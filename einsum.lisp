(in-package :clvt)

(defvar *vt-einsum-parse-cache* (make-hash-table :test 'equal))

(declaim (inline get-parsed-subscripts))
(defun get-parsed-subscripts (str)
  (declare (optimize (speed 3) (safety 0)) (simple-string str))
  (let ((cached (gethash str *vt-einsum-parse-cache*)))
    (if cached
        (values (the list (first cached))
		(the list (second cached))
		(the boolean (third cached)))
        (multiple-value-bind (inputs output explicit-p)
	    (parse-subscript-tokens str)
          (setf (gethash str *vt-einsum-parse-cache*)
		(list inputs output explicit-p))
          (values inputs output explicit-p)))))

(declaim (inline parse-subscript-tokens))
(defun parse-subscript-tokens (str)
  (declare (optimize (speed 3)) (simple-string str))
  (let ((len (length str))
	(inputs nil)
	(current-sub nil)
	(output nil)
	(i 0)
	(state :inputs))
    (declare (fixnum len i))
    (flet ((save-current-sub ()
             (when current-sub
               (let ((final-sub (nreverse current-sub)))
                 (if (eq state :inputs)
		     (push final-sub inputs)
		     (setf output final-sub)))
               (setf current-sub nil))))
      (loop while (< i len) do
        (let ((char (char str i)))
          (declare (character char))
          (cond
            ((and (char= char #\.) (< (+ i 2) len)
		  (char= (char str (+ i 1)) #\.)
		  (char= (char str (+ i 2)) #\.))
             (push :ellipsis current-sub)
	     (incf i 3))
            ((and (char= char #\-)
		  (< (1+ i) len)
		  (char= (char str (1+ i)) #\>))
             (save-current-sub)
	     (setf state :outputs)
	     (incf i 2))
            ((char= char #\,)
	     (when (eq state :inputs)
	       (save-current-sub))
	     (incf i))
            ((char= char #\space)
	     (incf i))
            ((char= char #\.)
	     (error "invalid syntax: single '.' found."))
            (t (push (the fixnum (char-code char))
		     current-sub)
	       (incf i)))))
      (save-current-sub)
      (values (nreverse inputs)
	      output
	      (if (eq state :outputs) t nil)))))

(declaim (inline expand-ellipsis))
(defun expand-ellipsis (input-subs output-sub vts)
  (with-float-safe
    (let ((ellipsis-ranks nil))
      (declare (list ellipsis-ranks))
      (loop for sub in input-subs for vt in vts
            for explicit-rank = (count-if #'(lambda (x)
					      (and (typep x 'fixnum)
						   (>= (the fixnum x) 0)))
					  sub)
            for has-ellipsis = (member :ellipsis sub)
            for tensor-rank = (length (vt-shape vt))
            for implicit-rank = (the fixnum
				     (if has-ellipsis
					 (- tensor-rank explicit-rank) 0))
            do (when (and has-ellipsis (< implicit-rank 0))
		 (error "subscript dimension mismatch"))
               (when (> (count :ellipsis sub) 1)
		 (error "only one ellipsis allowed"))
               (push implicit-rank ellipsis-ranks))
      (setf ellipsis-ranks (nreverse ellipsis-ranks))
      (let ((max-implicit-rank (reduce #'max ellipsis-ranks :initial-value 0)))
        (declare (fixnum max-implicit-rank))
        (let ((all-ellipsis-labels
		(loop for i from 1 to max-implicit-rank
		      collect (the fixnum (- i)))))
          (flet
	      ((expand-sub (sub implicit-rank)
                 (declare (list sub))
                 (let ((pos (position :ellipsis sub)))
                   (if (not pos) sub
                       (let* ((before (subseq sub 0 pos))
                              (after (nthcdr (1+ pos) sub))
                              (start-idx
				(the fixnum
				     (- max-implicit-rank implicit-rank)))
                              (labels-to-use
				(subseq all-ellipsis-labels start-idx)))
                         (nconc before labels-to-use after))))))
            (values (mapcar #'expand-sub input-subs ellipsis-ranks)
                    (when output-sub
		      (expand-sub output-sub max-implicit-rank)))))))))

(declaim (inline analyze-einsum))
(defun analyze-einsum (input-subs output-subs vts explicit-p)
  (declare (type list input-subs output-subs vts)
	   (type boolean explicit-p) (optimize (speed 3)))
  (with-float-safe
    (let ((label-dims
	    (make-array 266 :element-type 'fixnum :initial-element -1))
          (label-counts
	    (make-array 266 :element-type 'fixnum :initial-element 0))
          (all-labels-list nil))
      (macrolet
	  ((to-idx (code)
	     `(the fixnum (if (< ,code 0)
			      (+ 256 ,code)
			      ,code))))
        (loop for sub list in input-subs
	      for tns in vts for shape = (vt-shape tns) do
		(unless (= (length sub)
			   (length shape))
		  (error "subscript dimension mismatch"))
		(loop for label fixnum in sub
		      for dim fixnum in shape do
			(incf (aref label-counts (to-idx label)))
			(pushnew label all-labels-list)
			(let ((old-dim (aref label-dims (to-idx label))))
			  (when (and (/= old-dim -1)
				     (/= dim 1)
				     (/= old-dim 1)
				     (/= old-dim dim))
			    (error "dimension conflict for label ~a: ~a vs ~a" 
				   (if (< label 0)
				       (format nil "ellipsis-dim(~a)" label)
				       (string (code-char label))) old-dim dim))
			  (setf (aref label-dims (to-idx label))
				(cond ((= old-dim -1) dim)
				      ((or (= old-dim 0)
					   (= dim 0)) 0)
				      (t (max old-dim dim)))))))
        (let* ((final-output-subs output-subs)
	       (explicit-mode explicit-p))
          (unless explicit-mode
            (let ((ellipsis-subs nil)
		  (normal-subs nil))
              (loop for label fixnum in all-labels-list
                    for mapped-idx = (if (< label 0)
					 (+ 256 label)
					 label)
                    when (or (< label 0)
			     (= (aref label-counts mapped-idx) 1))
                      do (if (< label 0)
			     (push label ellipsis-subs)
			     (push label normal-subs)))
              (setf final-output-subs
		    (append (sort ellipsis-subs #'>)
			    (sort normal-subs #'<)))))
          (let* ((sum-labels
		   (set-difference all-labels-list final-output-subs))
                 (all-labels
		   (coerce (append final-output-subs sum-labels)
			   'simple-vector)))
            (values all-labels label-dims final-output-subs)))))))

;; 极速内核 1: double-float 
(declaim (inline %matmul-df-fast-kernel))
(defun %matmul-df-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (declare (type (simple-array double-float (*)) a-data b-data c-data)
           (type fixnum m k n a-off b-off c-off)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((a-ptr a-off)
        (c-row c-off)
        (n-main (the fixnum (logand n -4)))
        (n-rem  (the fixnum (logand n 3))))
    (declare (type fixnum a-ptr c-row n-main n-rem))
    (loop for i of-type fixnum from 0 below m do
      (loop for l of-type fixnum from 0 below k do
        (let ((a-val (aref a-data a-ptr))
              (b-ptr (the fixnum (+ b-off (the fixnum (* l n)))))
              (c-ptr c-row))
          (declare (type double-float a-val)
                   (type fixnum b-ptr c-ptr))
          (loop for j of-type fixnum from 0 below n-main by 4 do
            (let ((c0 (aref c-data c-ptr))
                  (c1 (aref c-data (the fixnum (1+ c-ptr))))
                  (c2 (aref c-data (the fixnum (+ c-ptr 2))))
                  (c3 (aref c-data (the fixnum (+ c-ptr 3))))
                  (b0 (aref b-data b-ptr))
                  (b1 (aref b-data (the fixnum (1+ b-ptr))))
                  (b2 (aref b-data (the fixnum (+ b-ptr 2))))
                  (b3 (aref b-data (the fixnum (+ b-ptr 3)))))
              (declare (type double-float c0 c1 c2 c3 b0 b1 b2 b3))
              (setf (aref c-data c-ptr)
		    (the double-float (+ c0 (the double-float (* a-val b0)))))
              (setf (aref c-data (the fixnum (1+ c-ptr)))
		    (the double-float (+ c1 (the double-float (* a-val b1)))))
              (setf (aref c-data (the fixnum (+ c-ptr 2)))
		    (the double-float (+ c2 (the double-float (* a-val b2)))))
              (setf (aref c-data (the fixnum (+ c-ptr 3)))
		    (the double-float (+ c3 (the double-float (* a-val b3))))))
            (incf c-ptr 4)
            (incf b-ptr 4))
          (loop for j of-type fixnum from 0 below n-rem do
            (let ((cv (aref c-data c-ptr))
                  (bv (aref b-data b-ptr)))
              (declare (type double-float cv bv))
              (setf (aref c-data c-ptr)
		    (the double-float (+ cv (the double-float (* a-val bv))))))
            (incf c-ptr)
            (incf b-ptr)))
        (incf a-ptr))
      (incf c-row n))))

;; 极速内核 2: int64
(declaim (inline %matmul-i64-fast-kernel))
(defun %matmul-i64-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (declare (type (simple-array (signed-byte 64) (*)) a-data b-data c-data)
           (type fixnum m k n a-off b-off c-off)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((a-ptr a-off)
        (c-ptr-base c-off))
    (declare (type fixnum a-ptr c-ptr-base))
    (loop for i of-type fixnum from 0 below m do
      (let ((b-ptr-base b-off))
        (declare (type fixnum b-ptr-base))
        (loop for l of-type fixnum from 0 below k do
          (let ((a-val (aref a-data a-ptr)))
            (declare (type (signed-byte 64) a-val))
            (let ((b-ptr b-ptr-base)
                  (c-ptr c-ptr-base))
              (declare (type fixnum b-ptr c-ptr))
              (loop for j of-type fixnum from 0 below n do
                (setf (aref c-data c-ptr)
                      (the (signed-byte 64)
                           (+ (the (signed-byte 64) (aref c-data c-ptr))
                              (the (signed-byte 64)
                                   (* a-val
                                      (the (signed-byte 64)
					   (aref b-data b-ptr)))))))
                (incf c-ptr)
                (incf b-ptr)))
            (incf b-ptr-base n))
          (incf a-ptr)))
      (incf c-ptr-base n))))

;; 极速内核 3: single-float (float32)
(declaim (inline %matmul-sf-fast-kernel))
(defun %matmul-sf-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (declare (type (simple-array single-float (*)) a-data b-data c-data)
           (type fixnum m k n a-off b-off c-off)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((a-ptr a-off)
        (c-row c-off)
        (n-main (the fixnum (logand n -4)))
        (n-rem  (the fixnum (logand n 3))))
    (declare (type fixnum a-ptr c-row n-main n-rem))
    (loop for i of-type fixnum from 0 below m do
      (loop for l of-type fixnum from 0 below k do
        (let ((a-val (aref a-data a-ptr))
              (b-ptr (the fixnum (+ b-off (the fixnum (* l n)))))
              (c-ptr c-row))
          (declare (type single-float a-val)
                   (type fixnum b-ptr c-ptr))
          ;; 4路循环展开
          (loop for j of-type fixnum from 0 below n-main by 4 do
            (let ((c0 (aref c-data c-ptr))
                  (c1 (aref c-data (the fixnum (1+ c-ptr))))
                  (c2 (aref c-data (the fixnum (+ c-ptr 2))))
                  (c3 (aref c-data (the fixnum (+ c-ptr 3))))
                  (b0 (aref b-data b-ptr))
                  (b1 (aref b-data (the fixnum (1+ b-ptr))))
                  (b2 (aref b-data (the fixnum (+ b-ptr 2))))
                  (b3 (aref b-data (the fixnum (+ b-ptr 3)))))
              (declare (type single-float c0 c1 c2 c3 b0 b1 b2 b3))
              (setf (aref c-data c-ptr)
		    (the single-float (+ c0 (the single-float (* a-val b0)))))
              (setf (aref c-data (the fixnum (1+ c-ptr)))
		    (the single-float (+ c1 (the single-float (* a-val b1)))))
              (setf (aref c-data (the fixnum (+ c-ptr 2)))
		    (the single-float (+ c2 (the single-float (* a-val b2)))))
              (setf (aref c-data (the fixnum (+ c-ptr 3)))
		    (the single-float (+ c3 (the single-float (* a-val b3))))))
            (incf c-ptr 4)
            (incf b-ptr 4))
          ;; 处理剩余不足4的尾部
          (loop for j of-type fixnum from 0 below n-rem do
            (let ((cv (aref c-data c-ptr))
                  (bv (aref b-data b-ptr)))
              (declare (type single-float cv bv))
              (setf (aref c-data c-ptr)
		    (the single-float (+ cv (the single-float (* a-val bv))))))
            (incf c-ptr)
            (incf b-ptr)))
        (incf a-ptr))
      (incf c-row n))))

;; 极速内核 4: int32 (signed-byte 32)
(declaim (inline %matmul-i32-fast-kernel))
(defun %matmul-i32-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (declare (type (simple-array (signed-byte 32) (*)) a-data b-data c-data)
           (type fixnum m k n a-off b-off c-off)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((a-ptr a-off)
        (c-ptr-base c-off))
    (declare (type fixnum a-ptr c-ptr-base))
    (loop for i of-type fixnum from 0 below m do
      (let ((b-ptr-base b-off))
        (declare (type fixnum b-ptr-base))
        (loop for l of-type fixnum from 0 below k do
          (let ((a-val (aref a-data a-ptr)))
            (declare (type (signed-byte 32) a-val))
            (let ((b-ptr b-ptr-base)
                  (c-ptr c-ptr-base))
              (declare (type fixnum b-ptr c-ptr))
              (loop for j of-type fixnum from 0 below n do
                (setf (aref c-data c-ptr)
                      (the (signed-byte 32)
                           (+ (the (signed-byte 32) (aref c-data c-ptr))
                              (the (signed-byte 32)
                                   (* a-val
                                      (the (signed-byte 32)
					   (aref b-data b-ptr)))))))
                (incf c-ptr)
                (incf b-ptr)))
            (incf b-ptr-base n))
          (incf a-ptr)))
      (incf c-ptr-base n))))

(defun einsum-execute
    (all-labels-vec label-dims-vec output-subs input-subs vts &key out)
  (declare (type simple-vector all-labels-vec)
           (type (simple-array fixnum (*)) label-dims-vec)
           (type list output-subs input-subs vts))
  (with-float-safe
    (let* ((rank (length all-labels-vec))
           (n-vts (length vts))
           (dims-vec (make-array rank :element-type 'fixnum)))
      
      (loop for i fixnum from 0 below rank
            for label fixnum = (svref all-labels-vec i) do
              (setf (aref dims-vec i)
		    (aref label-dims-vec
			  (if (< label 0) (+ 256 label) label))))
      
      (let* ((strides-mat (make-array (list n-vts rank)
                                      :element-type 'fixnum))
             (in-data-vec (make-array n-vts))
             (in-offsets-vec (make-array n-vts :element-type 'fixnum)))
        
        (loop for sub list in input-subs
              for tns in vts
              for t-idx fixnum from 0 do
		(let ((cont-tns (vt-contiguous tns)))
		  (let ((phys-strides (vt-strides cont-tns))
			(phys-shape (vt-shape cont-tns)))
		    (setf (aref in-data-vec t-idx) (vt-data cont-tns))
		    (setf (aref in-offsets-vec t-idx) (vt-offset cont-tns))
		    (let ((p-strides-arr (coerce phys-strides 'simple-vector))
			  (p-shape-arr (coerce phys-shape 'simple-vector)))
                      (loop for lbl fixnum across all-labels-vec
			    for logical-idx fixnum from 0 do
			      (let ((p-stride-sum 0))
				(declare (fixnum p-stride-sum))
				(loop for pos fixnum from 0 below (length sub)
				      for sub-lbl fixnum = (nth pos sub)
				      when (eql sub-lbl lbl) do
					(let ((p-dim (the fixnum (svref p-shape-arr pos)))
					      (p-stride (the fixnum
							     (svref p-strides-arr pos))))
					  (incf p-stride-sum
						(if (and (= p-dim 1)
							 (> (aref dims-vec logical-idx) 1))
						    0 p-stride)))) 
				(setf (aref strides-mat t-idx logical-idx)
				      p-stride-sum)
				))))))
        
        ;; 1. 扩展类型判断：支持 4 种数据类型
        (let* ((all-f64-p (every #'(lambda (vt) (eq (vt-dtype vt) :float64)) vts))
               (all-f32-p (every #'(lambda (vt) (eq (vt-dtype vt) :float32)) vts))
               (all-i64-p (every #'(lambda (vt) (eq (vt-dtype vt) :int64)) vts))
               (all-i32-p (every #'(lambda (vt) (eq (vt-dtype vt) :int32)) vts))
               (out-dtype (cond (all-f64-p :float64)
                                (all-f32-p :float32)
                                (all-i64-p :int64)
                                (all-i32-p :int32)
                                (t :float64)))
               (out-shape
		 (loop for lbl fixnum across all-labels-vec
                       when (member lbl output-subs)
                         collect (the fixnum (aref label-dims-vec
                                                   (if (< lbl 0)
                                                       (+ 256 lbl)
                                                       lbl))))))
          (declare (type boolean all-f64-p all-f32-p all-i64-p all-i32-p))
          
          (let* ((output (or out (vt-zeros out-shape :dtype out-dtype))))
            (when out
              (assert (equal (vt-shape output) out-shape)
                      (output) "out 形状不匹配!")
              (assert (eq (vt-dtype output) out-dtype)
                      (output) "out 类型不匹配!")
              (vt-fill output 0))
            
            (let* ((out-offset (vt-offset output))
                   (out-data (vt-data output))
                   (out-strides-vec
                     (make-array rank :element-type 'fixnum)))
              (declare (type fixnum out-offset))
              
              (let ((acc 1))
                (declare (fixnum acc))
                (loop for i fixnum from (1- rank) downto 0
                      for dim = (aref dims-vec i) do
			(if (member (svref all-labels-vec i) output-subs)
			    (progn
                              (setf (aref out-strides-vec i) acc)
                              (setf acc (the fixnum (* acc dim))))
			    (setf (aref out-strides-vec i) 0))))
              
              (when (or (zerop (vt-size output))
                        (some #'zerop out-shape)
                        (some #'zerop dims-vec))
                (return-from einsum-execute output))
              
              ;; 批量矩阵乘法 (BMM) 极速通道
              (when (and (= n-vts 2)
                         (>= (length output-subs) 2)
                         (or all-f64-p all-f32-p all-i64-p all-i32-p))
                (let* ((s1 (first input-subs))
                       (s2 (second input-subs))
                       (shared (intersection s1 s2))
                       (j-label (car (set-difference shared output-subs)))
                       (i-label (car (set-difference s1 s2)))
                       (k-label (car (set-difference s2 s1)))
                       (batch-labels (intersection shared output-subs)))
                  (when (and j-label i-label k-label
                             (= (length (set-difference shared output-subs)) 1)
                             (= (length (set-difference s1 s2)) 1)
                             (= (length (set-difference s2 s1)) 1))
                    (let* ((pos-i (position i-label all-labels-vec :test #'eql))
                           (pos-j (position j-label all-labels-vec :test #'eql))
                           (pos-k (position k-label all-labels-vec :test #'eql))
                           (d-i (aref dims-vec pos-i))
                           (d-j (aref dims-vec pos-j))
                           (d-k (aref dims-vec pos-k))
                           (sa-i (aref strides-mat 0 pos-i))
                           (sa-j (aref strides-mat 0 pos-j))
                           (sb-j (aref strides-mat 1 pos-j))
                           (sb-k (aref strides-mat 1 pos-k))
                           (so-i (aref out-strides-vec pos-i))
                           (so-k (aref out-strides-vec pos-k))
                           (off-a-base (aref in-offsets-vec 0))
                           (off-b-base (aref in-offsets-vec 1))
                           (data-a (aref in-data-vec 0))
                           (data-b (aref in-data-vec 1)))
                      
                      ;; 宏：生成 4 种类型的 BMM 内层循环，避免代码重复
                      (macrolet ((gen-bmm-logic (lisp-type kernel-name)
                                   `(let ((da (the (simple-array ,lisp-type (*)) data-a))
                                          (db (the (simple-array ,lisp-type (*)) data-b))
                                          (dc (the (simple-array ,lisp-type (*)) out-data)))
                                      (if (and (= sa-i d-j) (= sa-j 1) 
                                               (= sb-k 1) (= so-k 1)
                                               (= sb-j d-k) (= so-i d-k))
                                          ;; 极速通道
                                          (,kernel-name da db dc d-i d-j d-k off-a off-b off-c)
                                          ;; 兼容通道
                                          (loop for i-idx fixnum from 0 below d-i do
                                            (let ((ptr-a-row-start (+ off-a (the fixnum (* i-idx sa-i))))
                                                  (ptr-c-row-start (+ off-c (the fixnum (* i-idx so-i)))))
                                              (loop for j-idx fixnum from 0 below d-j do
                                                (let ((val-a (aref da (+ ptr-a-row-start (the fixnum (* j-idx sa-j)))))
                                                      (ptr-b-start (+ off-b (the fixnum (* j-idx sb-j))))
                                                      (ptr-c-start ptr-c-row-start))
                                                  (loop for k-idx fixnum from 0 below d-k do
                                                    (incf (aref dc ptr-c-start)
                                                          (* val-a (aref db ptr-b-start)))
                                                    (incf ptr-b-start sb-k)
                                                    (incf ptr-c-start so-k))))))))))
                        
                        (labels
                            ((loop-batch (b-labels off-a off-b off-c)
                               (if (null b-labels)
                                   (cond
                                     (all-f64-p (gen-bmm-logic double-float %matmul-df-fast-kernel))
                                     (all-f32-p (gen-bmm-logic single-float %matmul-sf-fast-kernel))
                                     (all-i64-p (gen-bmm-logic (signed-byte 64) %matmul-i64-fast-kernel))
                                     (all-i32-p (gen-bmm-logic (signed-byte 32) %matmul-i32-fast-kernel)))
                                   
                                   (let* ((lbl (first b-labels))
                                          (pos-lbl (position lbl all-labels-vec :test #'eql))
                                          (dim (aref dims-vec pos-lbl))
                                          (stride-a (aref strides-mat 0 pos-lbl))
                                          (stride-b (aref strides-mat 1 pos-lbl))
                                          (stride-c (aref out-strides-vec pos-lbl)))
                                     (loop for i fixnum from 0 below dim do
                                       (loop-batch (rest b-labels)
                                                   (+ off-a (* i stride-a))
                                                   (+ off-b (* i stride-b))
                                                   (+ off-c (* i stride-c))))))))
                          
                          (loop-batch batch-labels off-a-base off-b-base out-offset))
                        
                        (return-from einsum-execute output))))))
              
              ;; 通用 einsum 路径 (极简迭代状态机，零分配)
              (let ((cur-ptrs (make-array n-vts
                                          :element-type 'fixnum
                                          :initial-element 0)))
                (declare (type (simple-array fixnum (*)) cur-ptrs))
                (loop for k fixnum from 0 below n-vts do
                  (setf (aref cur-ptrs k) (aref in-offsets-vec k)))
                (let ((indices (make-array rank
                                           :element-type 'fixnum
                                           :initial-element 0))
                      (depth 0)
                      (out-ptr out-offset))
                  (declare (type (simple-array fixnum (*)) indices)
                           (type fixnum depth out-ptr))
                  (loop
                    (cond
                      ((= depth rank)
                       ;; 扩展为 4 种类型的强类型计算
                       (cond
                         (all-f64-p
                          (let ((product 1.0d0))
                            (declare (type double-float product))
                            (if (= n-vts 2)
                                (setf product
                                      (* (aref (the (simple-array double-float (*))
						    (aref in-data-vec 0))
					       (aref cur-ptrs 0))
                                         (aref (the (simple-array double-float (*))
						    (aref in-data-vec 1))
					       (aref cur-ptrs 1))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product (* product (aref (the (simple-array double-float (*))
								      (aref in-data-vec k))
								 (aref cur-ptrs k))))))
                            (incf (the double-float
				       (aref (the (simple-array double-float (*)) out-data) out-ptr))
				  product)))
                         
                         (all-f32-p
                          (let ((product 1.0f0))
                            (declare (type single-float product))
                            (if (= n-vts 2)
                                (setf product
                                      (* (aref (the (simple-array single-float (*))
						    (aref in-data-vec 0))
					       (aref cur-ptrs 0))
                                         (aref (the (simple-array single-float (*))
						    (aref in-data-vec 1))
					       (aref cur-ptrs 1))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product
					(* product (aref (the (simple-array single-float (*))
							      (aref in-data-vec k))
							 (aref cur-ptrs k))))))
                            (incf (the single-float
				       (aref (the (simple-array single-float (*))
						  out-data)
					     out-ptr))
				  product)))
                         
                         (all-i64-p
                          (let ((product 1))
                            (declare (type (signed-byte 64) product))
                            (if (= n-vts 2)
                                (setf product
                                      (* (the (signed-byte 64)
					      (aref (the (simple-array (signed-byte 64) (*))
							 (aref in-data-vec 0))
						    (aref cur-ptrs 0)))
                                         (the (signed-byte 64)
					      (aref (the (simple-array (signed-byte 64) (*))
							 (aref in-data-vec 1))
						    (aref cur-ptrs 1)))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product
					(the (signed-byte 64)
					     (* product (the (signed-byte 64)
							     (aref (the (simple-array (signed-byte 64) (*))
									(aref in-data-vec k))
								   (aref cur-ptrs k))))))))
                            (incf (the (signed-byte 64)
				       (aref (the (simple-array (signed-byte 64) (*)) out-data)
					     out-ptr))
				  product)))
                         
                         (all-i32-p
                          (let ((product 1))
                            (declare (type (signed-byte 32) product))
                            (if (= n-vts 2)
                                (setf product
                                      (* (the (signed-byte 32)
					      (aref (the (simple-array (signed-byte 32) (*))
							 (aref in-data-vec 0))
						    (aref cur-ptrs 0)))
                                         (the (signed-byte 32)
					      (aref (the (simple-array (signed-byte 32) (*))
							 (aref in-data-vec 1))
						    (aref cur-ptrs 1)))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product
					(the (signed-byte 32)
					     (* product
						(the (signed-byte 32)
						     (aref (the (simple-array (signed-byte 32) (*))
								(aref in-data-vec k))
							   (aref cur-ptrs k))))))))
                            (incf (the (signed-byte 32)
				       (aref (the (simple-array (signed-byte 32) (*))
						  out-data)
					     out-ptr))
				  product)))
                         
                         (t
                          (let ((product 1))
                            (if (= n-vts 2)
                                (setf product
                                      (* (aref (aref in-data-vec 0) (aref cur-ptrs 0))
                                         (aref (aref in-data-vec 1) (aref cur-ptrs 1))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product
					(* product (aref (aref in-data-vec k) (aref cur-ptrs k))))))
                            (incf (aref out-data out-ptr) product))))
                       
                       (loop
                         (decf depth)
                         (when (< depth 0)
                           (return-from einsum-execute output))
                         (let ((d depth))
                           (if (< (aref indices d) (1- (aref dims-vec d)))
                               (progn
                                 (incf out-ptr (aref out-strides-vec d))
                                 (loop for k fixnum from 0 below n-vts do
                                   (incf (aref cur-ptrs k) (aref strides-mat k d)))
                                 (incf (aref indices d))
                                 (incf depth)
                                 (return))
                               (progn
                                 (decf out-ptr
				       (the fixnum (* (aref indices d) (aref out-strides-vec d))))
                                 (loop for k fixnum from 0 below n-vts do
                                   (decf (aref cur-ptrs k)
					 (the fixnum (* (aref indices d) (aref strides-mat k d)))))
                                 (setf (aref indices d) 0))))))
                      (t (incf depth)))
		    ))))))))))


(defun vt-einsum (subscripts &rest args)
  "高性能 einsum (爱因斯坦求和约定) 终极接口。
  流程：
  1. 参数提取与清洗 (parse-vt-op-args)。
  2. 类型推导与统一 (astype)。
  3. 下标解析与语义分析。
  4. 直接调用 einsum-execute (自动触发 Matmul 极速内核)。"
  (declare (optimize (safety 0)))
  (multiple-value-bind (tensors dtype-arg out-arg)
      (parse-vt-op-args args)    
    (let* ((clean-tensors (mapcar #'ensure-vt tensors))
           (supported-types '(:float64 :float32 :int64 :int32)))
      ;; 校验显式 dtype
      (when (and dtype-arg (not (member dtype-arg supported-types)))
        (error "vt-einsum: 不支持的显式 dtype (~a)。允许: ~a。"
	       dtype-arg supported-types))
      ;; 校验 out 类型
      (when (and out-arg (not (member (vt-dtype out-arg) supported-types)))
        (error "vt-einsum: 不支持的 out 张量类型 (~a)。允许: ~a。"
	       (vt-dtype out-arg) supported-types))
      ;; 类型推导与统一 ===
      (let* ((final-dtype 
               (cond 
                 ;; 冲突检测：out 与 dtype 不一致时报错
                 ((and out-arg dtype-arg)
                  (unless (eq (vt-dtype out-arg) dtype-arg)
                    (error "vt-einsum: 类型冲突！:dtype (~a) 与 :out (~a) 不一致。"
                           dtype-arg (vt-dtype out-arg)))
                  (vt-dtype out-arg))
                 (out-arg  (vt-dtype out-arg))
                 (dtype-arg dtype-arg)
                 (t (apply #'vt-promote-type (mapcar #'vt-dtype clean-tensors)))))             
             ;; 类型转换 (零拷贝优化)
             (cast-tensors 
               (if (every #'(lambda (vt) (eq (vt-dtype vt) final-dtype)) clean-tensors)
                   clean-tensors
                   (mapcar #'(lambda (vt) (vt-astype vt final-dtype)) clean-tensors))))        
        (multiple-value-bind (raw-inputs raw-output explicit-p)
            (get-parsed-subscripts subscripts)          
          (unless (= (length raw-inputs) (length cast-tensors))
            (error "张量数量 ~a 与下标 ~a 不匹配" (length cast-tensors) raw-inputs))
          (multiple-value-bind (input-subs output-subs)
              (expand-ellipsis raw-inputs raw-output cast-tensors)
            (unless explicit-p (setf output-subs nil))            
            ;; 语义分析 (构建维度映射)
            (multiple-value-bind (all-labels label-dims output-subs-final)
                (analyze-einsum input-subs output-subs cast-tensors explicit-p)
              (einsum-execute 
               all-labels label-dims output-subs-final input-subs cast-tensors
               :out out-arg))))))))

(defun vt-matmul (a b &key dtype out)
  "矩阵乘法，兼容 1d 向量（对标 numpy 的 @ 运算符）。"
  (let ((ra (vt-order a))
        (rb (vt-order b)))
    (cond
      ;; 2d @ 2d → 2d（矩阵乘法）
      ((and (= ra 2) (= rb 2))
       (vt-einsum "ij,jk->ik" a b :dtype dtype :out out))
      ;; 1d @ 1d → 标量（内积）
      ((and (= ra 1) (= rb 1))
       (vt-einsum "i,i->" a b :dtype dtype :out out))
      ;; 2d @ 1d → 1d（矩阵乘向量）
      ((and (= ra 2) (= rb 1))
       (vt-einsum "ij,j->i" a b :dtype dtype :out out))
      ;; 1d @ 2d → 1d（向量乘矩阵）
      ((and (= ra 1) (= rb 2))
       (vt-einsum "i,ij->j" a b :dtype dtype :out out))
      ;; >2d @ >2d → 批量矩阵乘法
      (t (vt-einsum "...ij,...jk->...ik" a b :dtype dtype :out out)))))

(defun vt-@ (vt1 vt2 &key dtype out)
  (vt-matmul vt1 vt2 :dtype dtype :out out))
