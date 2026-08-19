;;;; io.lisp — 序列/数组与张量互转、打印

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 展平与嵌套
;;; ------------------------------------------------------------------

(defun vt-flatten-sequence (seq)
  "深度优先遍历 seq 及其嵌套序列，返回所有原子元素的列表（行主序）。"
  (with-float-safe
    (labels ((sequence-p (obj) (or (listp obj) (arrayp obj))))
      (if (not (sequence-p seq))
          (list seq)
          (let ((result '())
                (stack (list (cons seq (if (listp seq) seq 0)))))
            (loop
              (unless stack (return))
              (let* ((frame (pop stack))
                     (s (car frame))
                     (state (cdr frame)))
                (cond
                  ((listp s)
                   (when state
                     (let ((elt (car state)) (new-state (cdr state)))
                       (push (cons s new-state) stack)
                       (if (sequence-p elt)
                           (push (cons elt (if (listp elt) elt 0)) stack)
                           (push elt result)))))
                  ((arrayp s)
                   (let ((len (array-total-size s)))
                     (when (< state len)
                       (let ((elt (row-major-aref s state)))
                         (push (cons s (1+ state)) stack)
                         (if (sequence-p elt)
                             (push (cons elt (if (listp elt) elt 0)) stack)
                             (push elt result))))))
                  (t (error "~s 不是序列" s)))))
            (nreverse result))))))

(defun vt-from-sequence (contents &key (dtype :float64))
  "从嵌套序列创建张量（行主序）。支持任意维度规则嵌套；空序列 -> 形状 (0)。"
  (with-float-safe
    (labels ((infer-shape (seq)
               (typecase seq
                 (list
                  (if (null seq)
                      (list 0)
                      (let* ((first (car seq))
                             (rest-shape (typecase first
                                           (list (infer-shape first))
                                           (vector (infer-shape first))
                                           (t nil))))
                        (if rest-shape
                            (cons (length seq)
                                  (loop for sub in (cdr seq)
                                        unless (equal (infer-shape sub) rest-shape)
                                          do (error "不规则嵌套")
                                        finally (return rest-shape)))
                            (progn
                              (loop for sub in (cdr seq)
                                    when (or (listp sub) (typep sub 'vector))
                                      do (error "不规则嵌套"))
                              (list (length seq)))))))
                 (vector
                  (let ((len (length seq)))
                    (if (zerop len) (list 0)
                        (let* ((first (aref seq 0))
                               (rest-shape (typecase first
                                             (list (infer-shape first))
                                             (vector (infer-shape first))
                                             (t nil))))
                          (if rest-shape
                              (cons len
                                    (loop for i from 1 below len
                                          for sub = (aref seq i)
                                          unless (equal (infer-shape sub) rest-shape)
                                            do (error "不规则嵌套")
                                          finally (return rest-shape)))
                              (progn
                                (loop for i from 1 below len
                                      for sub = (aref seq i)
                                      when (or (listp sub) (typep sub 'vector))
                                        do (error "不规则嵌套"))
                                (list len)))))))
                 (t (error "无法从 ~s 创建张量" seq))))
             (fill-tensor (data seq shape strides flat-idx)
               (if (null shape)
                   (setf (aref data flat-idx)
                         (coerce seq (vt-dtype->lisp-type dtype)))
                   (let ((stride (first strides)) (current flat-idx))
                     (typecase seq
                       (list
                        (dolist (elem seq)
                          (fill-tensor data elem (rest shape) (rest strides) current)
                          (incf current stride)))
                       (vector
                        (loop for elem across seq do
                          (fill-tensor data elem (rest shape) (rest strides) current)
                          (incf current stride)))
                       (t (error "fill-tensor: 不支持的序列类型")))))))
      (let* ((shape (infer-shape contents))
             (size (vt-shape-to-size shape))
             (lisp-type (vt-dtype->lisp-type dtype))
             (data (make-array size :element-type lisp-type
                                     :initial-element (coerce 0 lisp-type)))
             (strides (vt-compute-strides shape)))
        (fill-tensor data contents shape strides 0)
        (%make-vt :data data :shape shape :strides strides :offset 0 :dtype dtype)))))

(defun vt-flatten-to-nested (dims data)
  "将行主序一维 data 转换为符合 dims 的嵌套列表。"
  (let ((idx 0))
    (labels ((recurse (dims)
               (if (null dims)
                   (prog1 (aref data idx) (incf idx))
                   (let ((n (first dims)) (result nil))
                     (dotimes (i n)
                       (declare (fixnum i))
                       (push (recurse (rest dims)) result))
                     (nreverse result)))))
      (recurse dims))))

(defun vt-to-list (vt)
  "将张量转换为嵌套列表，正确处理任意 strides/offset 的视图。"
  (labels ((build (shape strides offset data)
             (if (null shape)
                 (aref data offset)
                 (let ((dim (first shape)) (stride (first strides)) (result nil))
                   (loop for i fixnum from (1- dim) downto 0
                         for sub = (+ offset (* i stride))
                         do (push (build (rest shape) (rest strides) sub data) result))
                   result))))
    (let ((shape (vt-shape vt)) (strides (vt-strides vt))
          (offset (vt-offset vt)) (data (vt-data vt)))
      (if shape
          (build shape strides offset data)
          (aref data offset)))))

(defun vt-to-array (vt &key dtype)
  "将张量转换为原生多维数组。使用 vt-do-each 遍历，同时维护逻辑坐标。"
  (when dtype (setf vt (vt-astype vt dtype)))
  (let ((shape (vt-shape vt)))
    (if (null shape)
        ;; 标量：0 维数组
        (make-array nil :initial-element (aref (vt-data vt) (vt-offset vt)))
        ;; 非标量
        (let* ((rank (length shape))
               (dims (coerce shape 'simple-vector))          ; 各维度大小
               (arr (make-array shape :element-type (vt-element-type vt)))
               (coords (make-list rank :initial-element 0))) ; 初始坐标全 0
          (vt-do-each (ptr val vt)
            (declare (ignore val))
            ;; 使用当前坐标设置目标数组
            (setf (apply #'aref arr coords)
                  (aref (vt-data vt) ptr))
            ;; 更新坐标到下一个逻辑位置（C 顺序）
            (let ((i (1- rank)))
              (loop
                (incf (nth i coords))                ; 当前位 +1
                (when (< (nth i coords) (svref dims i))
                  (return))                           ; 未溢出，更新完成
                (setf (nth i coords) 0)              ; 溢出归零，进位
                (decf i)
                (when (< i 0) (return)))))           ; 所有位都溢出，遍历结束
          arr))))

;;; ------------------------------------------------------------------
;;; 打印
;;; ------------------------------------------------------------------

(defvar *vt-print-threshold* 3 "超过该数量后打印开始省略")
(defvar *vt-print-precision* 6 "浮点打印精度")
(defvar *vt-indent-step* 1 "缩进步长")

(defun %type-category (type)
  (cond ((or (eq type 'fixnum) (eq type 'integer) (eq type 'bit)
             (and (listp type) (member (first type) '(signed-byte unsigned-byte))))
         :integer)
        ((member type '(single-float double-float short-float long-float float))
         :float)
        (t :other)))

(defun %format-number (val type)
  (case (%type-category type)
    (:integer (format nil "~d" val))
    (:float
     (let* ((str (format nil "~,vf" *vt-print-precision* val))
            (trimmed (string-right-trim "0" str)))
       (when (and (> (length trimmed) 0)
                  (char= (char trimmed (1- (length trimmed))) #\.))
         (setf trimmed (concatenate 'string trimmed "0")))
       trimmed))
    (otherwise (format nil "~a" val))))

(defun %phys-idx (vt indices)
  (loop with strides = (vt-strides vt) with offset = (vt-offset vt)
        for idx in indices for stride in strides
        sum (* idx stride) into res
        finally (return (+ res offset))))

(defun print-vt-recursive (vt axis current-indices base-indent col-width element-type stream)
  (let* ((shape (vt-shape vt))
         (rank (length shape))
         (dim-size (nth axis shape))
         (is-last-axis (= axis (1- rank)))
         (truncated-p (> dim-size (* 2 *vt-print-threshold*)))
         (edge *vt-print-threshold*)
         (current-level-indent (+ base-indent (* (1+ axis) *vt-indent-step*))))
    (write-char #\[ stream)
    (flet ((print-item (idx)
             (if is-last-axis
                 (let* ((phys-idx (%phys-idx vt (append current-indices (list idx))))
                        (val (aref (vt-data vt) phys-idx))
                        (str (%format-number val element-type)))
                   (format stream "~v@a" col-width str))
                 (print-vt-recursive vt (1+ axis) (append current-indices (list idx))
                                     base-indent col-width element-type stream))))
      (cond
        ((not truncated-p)
         (loop for i from 0 below dim-size
               when (> i 0) do (if is-last-axis
                                   (write-string ", " stream)
                                   (format stream ",~%~v@a" current-level-indent ""))
               do (print-item i)))
        (t
         (loop for i from 0 below edge
               when (> i 0) do (if is-last-axis
                                   (write-string ", " stream)
                                   (format stream ",~%~v@a" current-level-indent ""))
               do (print-item i))
         (if is-last-axis
             (format stream ", ...")
             (format stream ",~%~v@a..." current-level-indent ""))
         (loop for i from (- dim-size edge) below dim-size
               do (progn (if is-last-axis
                             (write-string ", " stream)
                             (format stream ",~%~v@a" current-level-indent ""))
                         (print-item i))))))
    (write-char #\] stream)))

(defmethod print-object ((obj vt) stream)
  (print-unreadable-object (obj stream :type t :identity nil)
    (let ((shape (vt-shape obj)) (element-type (vt-element-type obj)))
      (format stream "shape:~a dtype:~a " shape element-type)
      (cond
        ((and shape (zerop (reduce #'* shape :initial-value 1)))
         (format stream "[] (empty)"))
        ((null shape)
         (format stream "~a" (%format-number (aref (vt-data obj) (vt-offset obj)) element-type)))
        (t
         (let ((max-width 0))
           (labels ((scan-visible (current-idxs axis)
                      (let ((dim (nth axis shape)) (is-last (= axis (1- (length shape)))))
                        (if is-last
                            (loop for i from 0 below (min dim (* 2 *vt-print-threshold*))
                                  for phys = (%phys-idx obj (append current-idxs (list i)))
                                  for w = (length (%format-number (aref (vt-data obj) phys) element-type))
                                  do (setf max-width (max max-width w)))
                            (loop for i from 0 below (min dim *vt-print-threshold*)
                                  do (scan-visible (append current-idxs (list i)) (1+ axis)))))))
             (scan-visible nil 0))
           (incf max-width 1)
           (fresh-line stream)
           (format stream "  ")
           (print-vt-recursive obj 0 nil 2 max-width element-type stream)))))))

(defun vt-set-print-options (&key threshold precision indent-step)
  (when threshold (setf *vt-print-threshold* threshold))
  (when precision (setf *vt-print-precision* precision))
  (when indent-step (setf *vt-indent-step* indent-step))
  (values))

(defun vt-get-print-options ()
  (list *vt-print-threshold* *vt-print-precision* *vt-indent-step*))
