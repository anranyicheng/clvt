(in-package :clvt)
;; 全局控制变量保持不变
(defvar *vt-print-threshold* 4)
(defvar *vt-print-precision* 4)
(defvar *vt-indent-step* 1)

;; 辅助函数：判断类型类别
(defun get-type-category (type)
  "将具体的 Lisp 类型归类为 :integer, :float 或 :other"
  (cond
    ;; 整数类型判断 (包括 fixnum, bit, 以及 signed/unsigned byte)
    ((or (eq type 'fixnum)
         (eq type 'integer)
         (eq type 'bit)
         (and (listp type)
	      (member (first type) '(signed-byte unsigned-byte))))
     :integer)    
    ;; 浮点类型判断
    ((member type '(single-float double-float short-float long-float float))
     :float)    
    ;; 其他类型 (如 T, character, complex 等)
    (t :other)))

(defun format-number-string (val type)
  "根据数据类型格式化数值字符串"
  (case (get-type-category type)
    (:integer
     ;; 整数：直接打印,无小数点
     (format nil "~d" val))    
    (:float
     ;; 浮点数：保留精度,去除末尾多余的0
     (let* ((str (format nil "~,vf" *vt-print-precision* val))
            (trimmed (string-right-trim "0" str)))
       ;; 如果去掉0后以小数点结尾,补一个0 (如 "1." -> "1.0")
       (when (and (> (length trimmed) 0) 
                  (char= (char trimmed (1- (length trimmed))) #\.))
         (setf trimmed (concatenate 'string trimmed "0")))
       trimmed))    
    (otherwise
     ;; 其他类型：默认打印
     (format nil "~a" val))))

;; 辅助函数保持不变
(defun calc-phys-idx (vt indices)
  (loop with strides = (vt-strides vt)
        with offset = (vt-offset vt)
        for idx in indices
        for stride in strides
        sum (* idx stride) into res
        finally (return (+ res offset))))


(defun print-vt-recursive
    (vt axis current-indices base-indent col-width element-type stream)
  "递归打印核心逻辑.
   element-type: 数据类型,传递给格式化函数."
  (let* ((shape (vt-shape vt))
         (rank (length shape))
         (dim-size (nth axis shape))
         (is-last-axis (= axis (1- rank)))
         (truncated-p (> dim-size (* 2 *vt-print-threshold*)))
         (edge *vt-print-threshold*)
         (current-level-indent
	   (+ base-indent (* (1+ axis) *vt-indent-step*))))
    (write-char #\[ stream)
    (flet ((print-item (idx)
             (if is-last-axis
                 (let* ((phys-idx (calc-phys-idx
				   vt (append current-indices (list idx))))
                        (val (aref (vt-data vt) phys-idx))
                        ;; 使用传入的类型进行格式化
                        (str (format-number-string val element-type)))
                   (format stream "~v@a" col-width str))
                 ;; 递归传递类型
                 (print-vt-recursive vt
                                     (1+ axis)
                                     (append current-indices (list idx))
                                     base-indent
                                     col-width
                                     element-type
                                     stream))))
      
      (cond
        ((not truncated-p)
         (loop
	   for i from 0 below dim-size
           when (> i 0) do
             (if is-last-axis
                 (write-string ", " stream)
                 (format stream ",~%~v@a" current-level-indent ""))
           do (print-item i)))        
        (t
         (loop
	   for i from 0 below edge
           when (> i 0) do
             (if is-last-axis
                 (write-string ", " stream)
                 (format stream ",~%~v@a" current-level-indent ""))
           do (print-item i))         
         (if is-last-axis
             (format stream ", ...")
             (format stream ",~%~v@a..." current-level-indent ""))
         (loop
	   for i from (- dim-size edge) below dim-size
           do (progn
                (if is-last-axis
                    (write-string ", " stream)
                    (format stream ",~%~v@a" current-level-indent ""))
                (print-item i))))))
    
    (write-char #\] stream)))

(defmethod print-object ((obj vt) stream)
  (print-unreadable-object (obj stream :type t :identity nil)
    (let ((shape (vt-shape obj))
          ;; 获取数据类型
          (element-type (vt-element-type obj)))
      ;; 打印头部：包含 shape 和 dtype
      (format stream "shape:~A dtype:~A " shape element-type)
      
      (cond
        ((zerop (reduce #'* shape :initial-value 1))
         (format stream "[] (empty)"))
        
        ((null shape)
         ;; 标量打印
         (format stream "~a" (format-number-string
			      (aref (vt-data obj)
				    (vt-offset obj))
			      element-type)))
        
        (t
         ;; 预计算最大宽度
         (let ((max-width 0))
           (labels
	       ((scan-visible (current-idxs axis)
                  (let ((dim (nth axis shape))
                        (is-last (= axis (1- (length shape)))))
                    (if is-last
                        (loop
			  for i from 0 below (min dim
						  (* 2 *vt-print-threshold*))
                          for phys = (calc-phys-idx
				      obj (append current-idxs (list i)))
                          ;; 计算宽度时使用对应类型格式
                          for w = (length (format-number-string
					   (aref (vt-data obj) phys)
					   element-type))
                          do (setf max-width (max max-width w)))
                        (loop
			  for i from 0 below (min dim *vt-print-threshold*)
                          do (scan-visible
			      (append current-idxs (list i)) (1+ axis)))))))
             (scan-visible nil 0))
           (incf max-width 1) 
           
           (fresh-line stream)
           (format stream "  ") 
           ;; 传递 element-type 给递归函数
           (print-vt-recursive
	    obj 0 nil 2 max-width element-type stream)))))))
