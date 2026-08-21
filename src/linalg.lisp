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
  (declare (optimize (speed 3) (safety 0)))
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
                 (declare (list sub) (fixnum implicit-rank))
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
           (type list output-subs input-subs vts)
	   (optimize (speed 3)))
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
          (declare (type boolean all-f64-p all-f32-p all-i64-p all-i32-p)
		   (list out-shape))
          
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
                                                   (the fixnum (+ off-a (the fixnum (* i stride-a))))
                                                   (the fixnum (+ off-b (the fixnum (* i stride-b))))
                                                   (the fixnum (+ off-c (the fixnum (* i stride-c))))))))))
                          
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
  (declare (optimize (speed 3) (safety 0)))
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
(in-package :clvt)

(defun vt-dot (a b &key dtype out) 
  "点积/内积，支持任意维度： 
   - 若 a,b 均为 1d → 向量内积，返回 0 维张量。 
   - 若 a 为 2d, b 为 1d → 矩阵乘向量，返回 1d 向量。 
   - 若 a 为 1d, b 为 2d → 向量乘矩阵，返回 1d 向量。 
   - 若 a,b 均为 2d → 矩阵乘法 a @ b。 
   - 若 a,b 秩均 ≥2 → 批量矩阵乘法 '...ij,...jk->...ik'。 
   其他情况请直接使用 vt-einsum。"
  (with-float-safe
    (let ((ra (length (vt-shape (ensure-vt a)))) 
          (rb (length (vt-shape (ensure-vt b)))))
      (cond ((and (= ra 0) (= rb 0)) 
             (vt-* a b :dtype dtype :out out))
            ((and (= ra 1) (= rb 1)) 
             (vt-einsum "i,i->" a b :dtype dtype :out out)) 
            ((and (= ra 2) (= rb 1)) 
             (vt-einsum "ij,j->i" a b :dtype dtype :out out)) 
            ((and (= ra 1) (= rb 2)) 
             (vt-einsum "i,ij->j" a b :dtype dtype :out out)) 
            ((and (= ra 2) (= rb 2)) 
             (vt-einsum "ij,jk->ik" a b :dtype dtype :out out)) 
            ((and (>= ra 2) (>= rb 2)) 
             (vt-einsum "...ij,...jk->...ik" a b :dtype dtype :out out)) 
            (t (error "vt-dot: unsupported dimensions (a: ~d, b: ~d).
                      use vt-einsum directly." ra rb))))))

(defun vt-outer (a b &key (flatten t) dtype out)
  "计算张量外积。
  flatten = t (默认):
    先将输入展平为一维向量，再计算外积，返回二维矩阵。
    完全兼容 numpy 的 outer 函数 (支持任意维度输入自动展平)。
  flatten = nil:
    保留输入的每个轴，将所有轴拼接形成新张量。
    例如：2d 与 3d 计算后得到 5d 张量；若包含标量(0维)，则直接广播相乘。"
  (with-float-safe
    (let* ((a-vt (ensure-vt a))
           (b-vt (ensure-vt b))
           (shape-a (vt-shape a-vt))
           (shape-b (vt-shape b-vt)))
      (if flatten
          (let ((1d-a (if (null shape-a) 
                          (vt-reshape a-vt '(1)) 
                          (vt-flatten a-vt)))
                (1d-b (if (null shape-b) 
                          (vt-reshape b-vt '(1)) 
                          (vt-flatten b-vt))))
            (vt-einsum "i,j->ij" 1d-a 1d-b :dtype dtype :out out))          
          (if (or (null shape-a) (null shape-b))
              (vt-* a-vt b-vt :dtype dtype :out out)
              (let* ((rank-a (length shape-a))
                     (rank-b (length shape-b))
                     (a-reshaped
                       (vt-reshape a-vt
                                   (append shape-a
                                           (make-list rank-b 
                                                      :initial-element 1))))
                     (b-reshaped
                       (vt-reshape b-vt
                                   (append (make-list rank-a 
                                                      :initial-element 1)
                                           shape-b))))
                (vt-* a-reshaped b-reshaped :dtype dtype :out out)))))))


(defun vt-trace (matrix &key dtype out)
  "矩阵迹: 对角线元素之和"
  (with-float-safe
    (vt-sum (vt-diagonal matrix) :dtype dtype :out out)))

(defun vt-norm (vt &key axis keepdims dtype out)
  "l2 范数 (欧几里得范数)
   优化: 如果提供 out，求和与开方将原地执行，零临时内存分配。"
  (with-float-safe
    (let ((sq (vt-square vt)))
      (if axis
          (let ((sum-res (vt-sum sq :axis axis :keepdims keepdims
                                 :dtype dtype :out out)))
            (vt-sqrt sum-res :dtype (vt-dtype sum-res) :out sum-res))
          (let ((sum-res (vt-sum sq :dtype dtype :out out)))
            (vt-sqrt sum-res :dtype (vt-dtype sum-res) :out sum-res))))))

(defun vt-l1-norm (vt &key axis keepdims dtype out)
  "l1 范数"
  (with-float-safe
    (vt-sum (vt-abs vt) :axis axis :keepdims keepdims
            :dtype dtype :out out)))

(defun vt-frobenius-norm (matrix &key axis keepdims dtype out)
  "frobenius 范数 (专用于矩阵)"
  (with-float-safe
    (vt-norm matrix :axis axis :keepdims keepdims :dtype dtype :out out)))

;;; 方程求解与矩阵分析

(defun ensure-contiguous-2d-vt (vt)
  "确保矩阵在内存中是连续的."
  (with-float-safe
    (if (vt-contiguous-p vt)
	vt
	(vt-contiguous vt))))

;;; 部分选主元 lu 分解 (返回 p, l, u，使 p*a = l*u)
(defun vt-lu (matrix)
  "lu 分解。返回，其中 p 为置换矩阵(由行交换向量表示)。支持非方阵。"
  (with-float-safe
    (let* ((a (ensure-contiguous-2d-vt (vt-astype matrix :float64)))
           (m (first (vt-shape a)))   ;; 行数
           (n (second (vt-shape a)))  ;; 列数
           (k-max (min m n))          ;; 最大消元步数
           (data (vt-data a))
           (s0 (first (vt-strides a))) ; 行步长
           (s1 (second (vt-strides a))) ; 列步长
           (off (vt-offset a))
           ;; 置换向量长度应等于行数
           (piv (loop for i from 0 below m collect i))
           (sign 1))
      (declare (type (simple-array double-float (*)) data)
               (type fixnum m n k-max s0 s1 off))
      (loop for k from 0 below k-max
            for max-row = k
            for max-val = (abs (aref data (+ off (* k s0) (* k s1))))
            do (loop for i from (1+ k) below m ;; 在当前列的下方行中寻找主元
                     for val = (abs (aref data (+ off (* i s0) (* k s1))))
                     when (> val max-val)
                     do (setf max-val val max-row i))
               (unless (zerop max-val)
                 ;; 交换行
                 (unless (= max-row k)
                   (rotatef (nth k piv) (nth max-row piv))
                   (setf sign (- sign))
                   (loop for j from 0 below n ;; 交换整行 (列数维度)
                         for ptr1 = (+ off (* k s0) (* j s1))
                         for ptr2 = (+ off (* max-row s0) (* j s1))
                         do (rotatef (aref data ptr1) (aref data ptr2))))
                 ;; 消元
                 (let ((pivot (aref data (+ off (* k s0) (* k s1)))))
                   (loop for i from (1+ k) below m ;; 消去当前列下方的行
                         for ptr-ik = (+ off (* i s0) (* k s1))
                         for multiplier = (/ (aref data ptr-ik) pivot)
                         do (setf (aref data ptr-ik) multiplier)
                            (loop for j from (1+ k) below n ;; 遍历右侧列
                                  for ptr-ij = (+ off (* i s0) (* j s1))
                                  for ptr-kj = (+ off (* k s0) (* j s1))
                                  do (decf (aref data ptr-ij)
                                           (* multiplier (aref data ptr-kj))))))))
      (values a piv sign))))


(defun vt-det (matrix &key out)
  "基于 lu 分解计算行列式，返回 0 维张量"
  (with-float-safe
    (let ((shape (vt-shape matrix)))
      ;; 校验方阵
      (unless (and (= (length shape) 2)
                   (= (first shape) (second shape)))
        (error "vt-det: 行列式仅支持方阵，收到形状 ~a" shape)))
    (multiple-value-bind (lu piv sign)
        (vt-lu matrix)
      (declare (ignore piv))
      (let* ((n (first (vt-shape lu)))
             (data (vt-data lu))
             (s0 (first (vt-strides lu)))
             (s1 (second (vt-strides lu)))
             (off (vt-offset lu))
             (det sign))
        (loop for i from 0 below n
              for pivot = (aref data (+ off (* i s0) (* i s1)))
              when (zerop pivot)
                do (return-from vt-det
                     (if out (vt-fill out 0.0d0)
			 (make-vt nil 0.0d0 :dtype :float64)))
              do (setf det (* det pivot)))
        (if out (vt-fill out det) (make-vt nil det :dtype :float64))))))


(defun vt-solve (a b &key out)
  "求解线性方程组 ax = b (支持多右端项)"
  (with-float-safe
    (let* ((a-shape (vt-shape a)))
      ;; 校验系数矩阵为方阵
      (unless (and (= (length a-shape) 2)
                   (= (first a-shape) (second a-shape)))
        (error "vt-solve: 线性方程组求解要求系数矩阵为方阵，收到形状 ~a" a-shape))
      (let* ((a (ensure-contiguous-2d-vt a))
             (b-vt (ensure-vt b))
             (n (first (vt-shape a)))
             (b-shape (vt-shape b-vt))
             (nrhs (if (> (length b-shape) 1) (second b-shape) 1))
             (b-copy (if (= nrhs 1)
			 (vt-reshape (vt-astype (vt-copy b-vt) :float64)
				     (list n 1))
			 (vt-astype (vt-copy b-vt) :float64)))
             (orig-b (vt-copy b-copy)))
	(multiple-value-bind (lu piv sign)
            (vt-lu a)
          (declare (ignore sign))
          (let ((lu-data (vt-data lu))
		(lu-s0 (first (vt-strides lu)))
		(lu-s1 (second (vt-strides lu)))
		(lu-off (vt-offset lu))
		(b-data (vt-data b-copy))
		(b-s0 (first (vt-strides b-copy)))
		(b-s1 (second (vt-strides b-copy)))
		(b-off (vt-offset b-copy))
		(ob-data (vt-data orig-b))
		(ob-s0 (first (vt-strides orig-b)))
		(ob-s1 (second (vt-strides orig-b)))
		(ob-off (vt-offset orig-b)))
            ;; 计算相对奇异阈值：基于 LU 矩阵的最大绝对值元素，
            ;; 避免绝对阈值 1e-12 对小范数矩阵误判为奇异、对大范数矩阵漏判。
            ;; 阈值 = n * eps * max|lu_ij|，n 为矩阵阶数，
            ;; eps = double-float-epsilon (≈2.22e-16)，n 倍补偿消元累积误差。
            (let ((max-abs 0.0d0))
              (declare (double-float max-abs))
              (loop for i below (length lu-data) do
                (let ((v (abs (aref lu-data i))))
                  (when (> v max-abs)
                    (setf max-abs v))))
              (let ((singular-threshold
                      (* (max 1 n) double-float-epsilon
                         (max max-abs 1.0d0))))
                (declare (double-float singular-threshold))
          ;; 1. 应用行置换 pb
            (loop for i from 0 below n do
              (loop for j from 0 below nrhs do
		(setf (aref b-data (+ b-off
				      (* i b-s0)
				      (* j b-s1)))
                      (aref ob-data (+ ob-off
				       (* (nth i piv) ob-s0)
				       (* j ob-s1))))))          
            ;; 2. 前代
            (loop for k from 0 below n do
              (loop for i from (1+ k) below n
                    for mult = (aref lu-data (+ lu-off
						(* i lu-s0)
						(* k lu-s1)))
                    do (loop for j from 0 below nrhs do
                      (decf (aref b-data (+ b-off
					    (* i b-s0)
					    (* j b-s1)))
                            (* mult (aref b-data (+ b-off
						    (* k b-s0)
						    (* j b-s1))))))))          
            ;; 3. 回代
            (loop for k from (1- n) downto 0 do
              (let ((pivot (aref lu-data (+ lu-off
					    (* k lu-s0)
					    (* k lu-s1)))))
		(when (or (zerop pivot)
			  (< (abs pivot) singular-threshold))
		  (error "LinAlgError: Singular matrix. Cannot solve or invert."))
		(loop for j from 0 below nrhs do
                  (setf (aref b-data (+ b-off
					(* k b-s0)
					(* j b-s1)))
			(/ (aref b-data (+ b-off
					   (* k b-s0)
					   (* j b-s1)))
			   pivot)))
		(loop for i from 0 below k
                      for factor = (aref lu-data (+ lu-off
						    (* i lu-s0)
						    (* k lu-s1)))
                      do (loop for j from 0 below nrhs do
			(decf (aref b-data (+ b-off
					      (* i b-s0)
					      (* j b-s1)))
                              (* factor (aref b-data (+ b-off
							(* k b-s0)
							(* j b-s1)))))))))
	    (let ((res (if (= nrhs 1)
                           (vt-reshape b-copy (list n))
                           b-copy)))
              (if out
                  (vt-map #'identity res :out out)
                  res))))))))))

(defun vt-inv (matrix)
  "矩阵求逆。"
  (with-float-safe
    (let* ((n (first (vt-shape matrix)))
           (identity (vt-eye n :dtype (vt-dtype matrix))))
      (vt-solve matrix identity))))


;;; 3. 矩阵分解

(defun vt-qr (matrix &key (mode :reduced))
  "矩阵 qr 分解。 
   matrix : m×n 矩阵。
   mode :reduced 返回 q(m×k), r(k×n)，k = min(m,n)。
         :full 返回 q(m×m), r(m×n)。
   返回。"
  (assert (= 2 (vt-order matrix)))
  (with-float-safe
    (let* ((row (first (vt-shape matrix)))
           (col (second (vt-shape matrix)))
           (k (min row col))
           (r (vt-astype matrix :float64))
           (vlist (make-array k :initial-element nil))
           (betas (make-array k :element-type 'double-float)))
      
      ;; ---- 1. 正向分解，更新 r ----
      (loop for i from 0 below k
            for x = (vt-slice r (list i row) (list i)) ;; 第 i 列，i 行开始
            do (if (<= (vt-size x) 1)
                   (setf (aref betas i) 0.0d0)
                   (multiple-value-bind (v beta sigma)
                       (compute-householder x)
                     (declare (ignore sigma))
                     (setf (aref vlist i) v
                           (aref betas i) beta)
                     ;; 对子矩阵 r(i:m, i:n) 应用反射: R = R - beta * v * (v^T R)
                     ;; 直接用循环实现，避免 einsum 的 stride 问题
                     (let* ((m-sub (- row i))
                            (n-sub (- col i))
                            (r-data (vt-data r))
                            (r-s0 (first (vt-strides r)))
                            (r-s1 (second (vt-strides r)))
                            (r-off (vt-offset r))
                            (v-data (vt-data v))
                            (v-stride (first (vt-strides v)))
                            (v-off (vt-offset v)))
                       ;; w[j] = sum_k v[k] * R[i+k, i+j]
                       (let ((w (make-array n-sub :element-type 'double-float :initial-element 0.0d0)))
                         (loop for j fixnum from 0 below n-sub do
                           (let ((s 0.0d0))
                             (loop for ii fixnum from 0 below m-sub do
                               (incf s (* (aref v-data (+ v-off (* ii v-stride)))
                                          (aref r-data (+ r-off (* (+ i ii) r-s0) (* (+ i j) r-s1))))))
                             (setf (aref w j) s)))
                         ;; R[i+ii, i+j] -= beta * v[ii] * w[j]
                         (loop for ii fixnum from 0 below m-sub do
                           (let ((vi (aref v-data (+ v-off (* ii v-stride)))))
                             (loop for j fixnum from 0 below n-sub do
                               (decf (aref r-data (+ r-off (* (+ i ii) r-s0) (* (+ i j) r-s1)))
                                     (* beta vi (aref w j)))))))))))
      
      ;; ---- 2. 反向累积 q (必须从 k-1 到 0) ----
      (let* ((need-full (eq mode :full))
             (q (if need-full
                    (vt-eye row :cols row :dtype :float64)
                    (vt-eye row :cols k :dtype :float64))))
        
        (loop for i from (1- k) downto 0 ;; 反向循环
              for beta = (aref betas i)
              for v = (aref vlist i)
              when (and v (> beta 0.0d0))
                do ;; Q[i:m, :] = Q[i:m, :] - beta * v * (v^T Q[i:m, :])
                   (let* ((m-sub (- row i))
                          (nq (second (vt-shape q)))
                          (q-data (vt-data q))
                          (q-s0 (first (vt-strides q)))
                          (q-s1 (second (vt-strides q)))
                          (q-off (vt-offset q))
                          (v-data (vt-data v))
                          (v-stride (first (vt-strides v)))
                          (v-off (vt-offset v)))
                     ;; w[j] = sum_k v[k] * Q[i+k, j]
                     (let ((w (make-array nq :element-type 'double-float :initial-element 0.0d0)))
                       (loop for j fixnum from 0 below nq do
                         (let ((s 0.0d0))
                           (loop for ii fixnum from 0 below m-sub do
                             (incf s (* (aref v-data (+ v-off (* ii v-stride)))
                                        (aref q-data (+ q-off (* (+ i ii) q-s0) (* j q-s1))))))
                           (setf (aref w j) s)))
                       ;; Q[i+ii, j] -= beta * v[ii] * w[j]
                       (loop for ii fixnum from 0 below m-sub do
                         (let ((vi (aref v-data (+ v-off (* ii v-stride)))))
                           (loop for j fixnum from 0 below nq do
                             (decf (aref q-data (+ q-off (* (+ i ii) q-s0) (* j q-s1)))
                                   (* beta vi (aref w j)))))))))
        
        (values q (if need-full r (vt-slice r (list 0 k) '(:all))))))))

(defun compute-householder (x)
  "给定向量 x，返回 v, beta, sigma 使得
   h = i - beta * v * v^t 满足 h*x = sigma * e1。
   其中 sigma = -sign(x[0]) * ||x||，β = 2 / ||v||²。
   当 x 为零向量时返回 beta=0, sigma=0。"
  (with-float-safe
    (let* ((x-data (vt-data x))
           (x-stride (first (vt-strides x)))
           (x-off (vt-offset x))
           (size (vt-size x))
           (norm-sq 0.0d0))
      (loop for i from 0 below size
            for ptr = (+ x-off (* i x-stride))
            for val = (aref x-data ptr)
            do (incf norm-sq (* val val)))
      (let ((norm (sqrt norm-sq)))
        (if (zerop norm)
            (values (vt-copy x) 0.0d0 0.0d0)
            (let* ((sx0 (aref x-data x-off))
                   (sigma (if (>= sx0 0.0d0) (- norm) norm))
                   (v (vt-copy x))
                   (v-data (vt-data v)))
              (setf (aref v-data 0) (- sx0 sigma))
              (let* ((beta-num 0.0d0)
                     (v-stride (first (vt-strides v)))
                     (v-off (vt-offset v)))
                (loop for i from 0 below size
                      for ptr = (+ v-off (* i v-stride))
                      for val = (aref v-data ptr)
                      do (incf beta-num (* val val)))
                (let ((beta (/ 2.0d0 beta-num)))
                  (values v beta sigma)))))))))

(defun vt-matrix-rank (matrix &optional (tol 1e-10))
  "计算矩阵的秩 (线性代数定义：线性无关的行/列数)。
   基于带部分选主元的高斯消元法，统计非零主元的数量。
   等价于 numpy.linalg.matrix_rank。"
  (assert (= 2 (length (vt-shape matrix))) 
          (matrix) "vt-matrix-rank requires a 2D tensor")
  (with-float-safe
    (let* ((m (first (vt-shape matrix)))
           (n (second (vt-shape matrix)))
           ;; 复制一份，避免破坏原矩阵
           (a (vt-astype matrix :float64))
           (a-data (vt-data a))
           (a-offset (vt-offset a))
           (s0 (first (vt-strides a))) ; 行步长
           (s1 (second (vt-strides a))); 列步长
           (rank 0)
           (row 0))
      ;; 从第一列到最后一列进行消元
      (loop for col from 0 below n
            while (< row m) do
        ;; 1. 在当前列及以下的行中，寻找绝对值最大的主元 (部分选主元)
        (let ((max-val 0.0d0)
              (max-row row))
          (loop for i from row below m
                for val = (abs (aref a-data (+ a-offset
					       (* i s0)
					       (* col s1))))
                when (> val max-val)
                do (setf max-val val max-row i))
          
          ;; 2. 判断主元是否足够大 (大于容差 tol)
          (if (> max-val tol)
              (progn
                (incf rank)
                ;; 3. 如果最大主元不在当前行，则交换行
                (unless (= max-row row)
                  (loop for j from col below n
                        for off1 = (+ a-offset (* row s0) (* j s1))
                        for off2 = (+ a-offset (* max-row s0) (* j s1))
                        do (rotatef (aref a-data off1) (aref a-data off2))))
                ;; 4. 消元：将当前列下方的元素清零
                (let ((pivot (aref a-data (+ a-offset
					     (* row s0)
					     (* col s1)))))
                  (loop for i from (1+ row) below m
                        for multiplier = (/ (aref a-data (+ a-offset (* i s0)
							    (* col s1)))
					    pivot)
                        do (loop for j from (1+ col) below n
                                 for off-target = (+ a-offset (* i s0) (* j s1))
                                 for off-source = (+ a-offset (* row s0) (* j s1))
                                 do (decf (aref a-data off-target)
					  (* multiplier (aref a-data off-source)))))
                  ;; 物理上将下方元素置零，保证数值干净
                  (loop for i from (1+ row) below m
                        do (setf (aref a-data (+ a-offset (* i s0)
						 (* col s1)))
				 0.0d0)))
                ;; 5. 处理下一行
                (incf row))
              ;; 如果主元太小，说明该列线性相关，跳过该列，继续看下一列
              nil)))
      rank)))

(defun extend-orthogonal-basis (u-econ &key (rng *vt-default-random-state*))
  "将 m×k 的 u_econ 通过随机向量 + gram-schmidt 补全为 m×m 正交矩阵。"
  (declare (random-state rng))
  (with-float-safe
    (let* ((m (first (vt-shape u-econ)))
           (k (second (vt-shape u-econ)))
           (extra (- m k)))
      (if (zerop extra) u-econ
          (let ((u-full (vt-zeros (list m m) :dtype :float64)))
            (dotimes (i k)
              (setf (vt-slice u-full '(:all) (list i))
                    (vt-slice u-econ '(:all) (list i))))
            (loop for col from k below m
                  for v = (vt-random (list m) :rng rng) do
                    (loop repeat 2 do
                      (dotimes (j col)
                        ;; flatten 避免 2D-1D 广播分配
                        (let* ((uj (vt-flatten
				    (vt-slice u-full '(:all) (list j))))
                               (proj (vt-ref (vt-dot uj v))))
                          (setf v (vt-- v (vt-scale uj proj))))))
                    (let ((norm (sqrt (vt-ref (vt-dot v v)))))
                      (if (> norm 1e-12)
                          (setf (vt-slice u-full '(:all) (list col))
                                (vt-scale v (/ 1.0 norm)))
                          (error "failed to generate orthogonal vector"))))
            u-full)))))


(defun vt-svd (matrix &key (full-matrices nil) (max-sweeps 50) (tol 1e-10))
  "奇异值分解 a = u s v^t。
  full-matrices : t 则返回完整尺寸 u(m×m), s(k), vt(n×n) (k = min(m,n))
                  : nil 返回经济尺寸 u(m×k), s(k), vt(k×n)
  max-sweeps : jacobi 最大扫描次数
  tol : 收敛容差"
  (assert (= 2 (vt-order matrix)))
  (with-float-safe
    (let* ((mat (vt-astype matrix :float64))
           (m (first (vt-shape mat)))
           (n (second (vt-shape mat)))
           (k (min m n)))
      (when (and (= m 1) (= n 1))
        (let ((val (aref (vt-data mat) (vt-offset mat))))
          (return-from vt-svd
            (values (vt-const '(1 1) 1.0d0 :dtype :float64)
                    (vt-const '(1) (abs val) :dtype :float64)
                    (if (>= val 0) (vt-ones '(1 1) :dtype :float64)
                        (vt-const '(1 1) -1.0d0 :dtype :float64))))))
      (let* ((u (vt-copy mat))
             (v (vt-eye n :dtype :float64))
             (changed t)
             (sweep 0))
        (let ((u-data (vt-data u))
              (u-s0 (first (vt-strides u)))
              (u-s1 (second (vt-strides u)))
              (u-off (vt-offset u))
              (v-data (vt-data v))
              (v-s0 (first (vt-strides v)))
              (v-s1 (second (vt-strides v)))
              (v-off (vt-offset v)))
          (flet ((col-norm-sq (col)
                   (let ((sum 0.0d0))
                     (loop for r from 0 below m
                           for ptr = (+ u-off (* r u-s0) (* col u-s1))
                           do (incf sum (expt (aref u-data ptr) 2)))
                     sum))
                 (col-dot (c1 c2)
                   (let ((sum 0.0d0))
                     (loop for r from 0 below m
                           for ptr1 = (+ u-off (* r u-s0) (* c1 u-s1))
                           for ptr2 = (+ u-off (* r u-s0) (* c2 u-s1))
                           do (incf sum (* (aref u-data ptr1)
                                           (aref u-data ptr2))))
                     sum)))
            (loop while (and changed (< sweep max-sweeps)) do
              (setf changed nil)
              (loop for i from 0 below (1- n) do
                (loop for j from (1+ i) below n
                      for alpha = (col-norm-sq i)
                      for beta = (col-norm-sq j)
                      for gamma = (col-dot i j)
                      when (> (abs gamma) (* tol (sqrt (* alpha beta)))) do
                        (setf changed t)
                        (let* ((zeta (/ (- beta alpha) (* 2 gamma)))
                               (t-abs (/ 1.0d0 (+ (abs zeta)
                                                 (sqrt (+ 1 (* zeta zeta))))))
                               (t-val (if (>= zeta 0) t-abs (- t-abs)))
                               (c (/ 1.0d0 (sqrt (+ 1 (* t-val t-val)))))
                               (s (* c t-val)))
                          (loop for r from 0 below m do
                            (let* ((ptr-i (+ u-off (* r u-s0) (* i u-s1)))
                                   (ptr-j (+ u-off (* r u-s0) (* j u-s1)))
                                   (ui (aref u-data ptr-i))
                                   (uj (aref u-data ptr-j)))
                              (setf (aref u-data ptr-i) (- (* ui c) (* uj s)))
                              (setf (aref u-data ptr-j) (+ (* ui s) (* uj c)))))
                          (loop for r from 0 below n do
                            (let* ((ptr-i (+ v-off (* r v-s0) (* i v-s1)))
                                   (ptr-j (+ v-off (* r v-s0) (* j v-s1)))
                                   (vi (aref v-data ptr-i))
                                   (vj (aref v-data ptr-j)))
                              (setf (aref v-data ptr-i) (- (* vi c) (* vj s)))
                              (setf (aref v-data ptr-j) (+ (* vi s) (* vj c))))))))
              (incf sweep))
            (let* ((s-vec (make-array k :element-type 'double-float))
                   (u-k (vt-zeros (list m k) :dtype :float64))
                   (pairs (sort (loop for col from 0 below n
                                      collect (cons (sqrt (col-norm-sq col)) col))
                                #'> :key #'car)))
              (dotimes (new-i k)
                (destructuring-bind (val . old-col) (nth new-i pairs)
                  (setf (aref s-vec new-i) val)
                  (setf (vt-slice u-k '(:all) (list new-i))
                        (vt-slice u '(:all) (list old-col)))))
              (dotimes (i k)
                (if (zerop (aref s-vec i))
                    (let ((random-v (vt-random (list m) :rng *vt-default-random-state*)))
                      (loop repeat 2 do
                        (dotimes (j i)
                          ;; flatten 避免 2D-1D 广播分配
                          (let* ((uj (vt-flatten (vt-slice u-k '(:all) (list j))))
                                 (proj (vt-ref (vt-dot uj random-v))))
                            (setf random-v (vt-- random-v (vt-scale uj proj))))))
                      (let ((norm (sqrt (vt-ref (vt-dot random-v random-v)))))
                        (if (> norm 1e-12)
                            (setf (vt-slice u-k '(:all) (list i))
                                  (vt-scale random-v (/ 1.0d0 norm)))
                            (setf (vt-slice u-k '(:all) (list i)) random-v))))
                    (let ((inv (/ 1.0d0 (aref s-vec i))))
                      (setf (vt-slice u-k '(:all) (list i))
                            (vt-scale (vt-slice u-k '(:all) (list i)) inv)))))
              (let* ((v-sorted (vt-zeros (list n n) :dtype :float64))
                     (used-cols nil))
                (dotimes (new-i k)
                  (let ((old-col (cdr (nth new-i pairs))))
                    (push old-col used-cols)
                    (setf (vt-slice v-sorted '(:all) (list new-i))
                          (vt-slice v '(:all) (list old-col)))))
                (let ((rest-cols (loop for col from 0 below n
                                       unless (member col used-cols) collect col)))
                  (loop for offset from 0 for col in rest-cols do
                    (setf (vt-slice v-sorted '(:all) (list (+ k offset)))
                          (vt-slice v '(:all) (list col)))))
                (let ((s-vt (vt-from-sequence (coerce s-vec 'list) :dtype :float64))
                      (vt (vt-transpose v-sorted)))
                  (if (not full-matrices)
                      (let ((vt-k (vt-slice vt (list 0 k) '(:all))))
                        (values u-k s-vt vt-k))
                      (cond ((= m n) (values u-k s-vt vt))
                            ((> m n) (let ((u-full (extend-orthogonal-basis u-k)))
                                       (values u-full s-vt vt)))
                            ((< m n) (values u-k s-vt vt))))))))))))) 

;;;; linalg-extensions.lisp — 扩展线性代数功能

(in-package :clvt)

;;; ============================================================
;;; 1. Cholesky 分解
;;; ============================================================

(defun vt-cholesky (matrix &key (upper nil))
  "对正定对称矩阵进行 Cholesky 分解。"
  (assert (= 2 (vt-order matrix)))
  (let ((shape (vt-shape matrix)))
    (assert (= (first shape) (second shape))
            (matrix) "Cholesky 分解要求方阵"))
  (with-float-safe
    (let* ((n (first (vt-shape matrix)))
           (a (vt-astype matrix :float64))
           (l (vt-zeros (list n n) :dtype :float64))
           (a-data (vt-data a))
           (l-data (vt-data l))
           (a-s0 (first (vt-strides a)))
           (a-s1 (second (vt-strides a)))
           (a-off (vt-offset a))
           (l-s0 (first (vt-strides l)))
           (l-s1 (second (vt-strides l)))
           (l-off (vt-offset l)))
      (declare (type (simple-array double-float (*)) a-data l-data)
               (type fixnum n a-s0 a-s1 a-off l-s0 l-s1 l-off))
      (loop for j fixnum from 0 below n do
        (let ((sum (aref a-data (+ a-off (* j a-s0) (* j a-s1)))))
          (declare (type double-float sum))
          (loop for k fixnum from 0 below j do
            (let ((l-jk (aref l-data (+ l-off (* j l-s0) (* k l-s1)))))
              (decf sum (* l-jk l-jk))))
          (when (<= sum 0.0d0)
            (error "vt-cholesky: 矩阵不是正定的"))
          (setf (aref l-data (+ l-off (* j l-s0) (* j l-s1))) (sqrt sum)))
        (loop for i fixnum from (1+ j) below n do
          (let ((sum (aref a-data (+ a-off (* i a-s0) (* j a-s1)))))
            (declare (type double-float sum))
            (loop for k fixnum from 0 below j do
              (decf sum (* (aref l-data (+ l-off (* i l-s0) (* k l-s1)))
                           (aref l-data (+ l-off (* j l-s0) (* k l-s1))))))
            (setf (aref l-data (+ l-off (* i l-s0) (* j l-s1)))
                  (/ sum (aref l-data (+ l-off (* j l-s0) (* j l-s1))))))))
      (if upper (vt-transpose l) l))))

;;; ============================================================
;;; 2. 对称矩阵特征值分解 (Jacobi 旋转法)
;;; ============================================================

(defun vt-eig (matrix &key (max-iter 200) (tol 1e-10))
  "对称矩阵特征值分解 (Jacobi 旋转法)。"
  (assert (= 2 (vt-order matrix)))
  (let ((shape (vt-shape matrix)))
    (assert (= (first shape) (second shape))
            (matrix) "特征值分解要求方阵"))
  (with-float-safe
    (let* ((n (first (vt-shape matrix)))
           (a (vt-copy (vt-astype matrix :float64)))
           (v (vt-eye n :dtype :float64)))
      ;; Jacobi iterations - use local variables to avoid compiler issues
      (loop for iter fixnum from 0 below max-iter do
        ;; Check convergence
        (let ((off-sum 0.0d0))
          (declare (type double-float off-sum))
          (loop for i fixnum from 0 below n do
            (loop for j fixnum from (1+ i) below n do
              (let ((val (vt-ref a i j)))
                (incf off-sum (* val val)))))
          (when (< (sqrt off-sum) tol) (return)))
        ;; Apply rotations
        (loop for p fixnum from 0 below (1- n) do
          (loop for q fixnum from (1+ p) below n do
            (let ((apq (vt-ref a p q)))
              (when (> (abs apq) tol)
                (let* ((app (vt-ref a p p))
                       (aqq (vt-ref a q q))
                       (theta (* 0.5d0 (atan (* 2.0d0 apq) (- app aqq))))
                       (c (cos theta))
                       (s (sin theta)))
                  ;; Update A
                  (setf (vt-ref a p q) 0.0d0)
                  (setf (vt-ref a q p) 0.0d0)
                  (setf (vt-ref a p p) (+ (* c c app) (* 2.0d0 s c apq) (* s s aqq)))
                  (setf (vt-ref a q q) (+ (* s s app) (* -2.0d0 s c apq) (* c c aqq)))
                  ;; Update off-diagonal
                  (loop for r fixnum from 0 below n
                        when (and (/= r p) (/= r q)) do
                          (let ((arp (vt-ref a r p))
                                (arq (vt-ref a r q)))
                            (let ((na (+ (* c arp) (* s arq)))
                                  (nb (+ (* (- s) arp) (* c arq))))
                              (setf (vt-ref a r p) na)
                              (setf (vt-ref a p r) na)
                              (setf (vt-ref a r q) nb)
                              (setf (vt-ref a q r) nb))))
                  ;; Update eigenvectors
                  (loop for r fixnum from 0 below n do
                    (let ((vp (vt-ref v r p))
                          (vq (vt-ref v r q)))
                      (setf (vt-ref v r p) (+ (* c vp) (* s vq)))
                      (setf (vt-ref v r q) (+ (* (- s) vp) (* c vq)))))))))))
      ;; Extract and sort eigenvalues
      (let ((pairs (loop for i fixnum from 0 below n
                         collect (cons (vt-ref a i i) i))))
        (setf pairs (sort pairs #'> :key #'car))
        (let ((vals (vt-from-sequence (mapcar #'car pairs) :dtype :float64))
              (vec (vt-zeros (list n n) :dtype :float64)))
          (loop for ni fixnum from 0 below n
                for oi = (cdr (nth ni pairs))
                do (loop for r fixnum from 0 below n do
                     (setf (vt-ref vec r ni) (vt-ref v r oi))))
          (values vals vec))))))


(defun vt-pinv (matrix &key (rcond 1e-15))
  "Moore-Penrose 伪逆。"
  (assert (= 2 (vt-order matrix)))
  (with-float-safe
    (multiple-value-bind (u s vt-mat) (vt-svd matrix)
      (let* ((s-data (vt-data s))
             (s-size (vt-size s))
             (s-max (loop for i fixnum from 0 below s-size maximize (aref s-data i)))
             (cutoff (* rcond s-max))
             (s-pinv (vt-zeros (list s-size) :dtype :float64)))
        (loop for i fixnum from 0 below s-size do
          (when (> (aref s-data i) cutoff)
            (setf (vt-ref s-pinv i) (/ 1.0d0 (aref s-data i)))))
        (let* ((n (second (vt-shape vt-mat)))
               (k (vt-size s))
               (v-scaled (vt-zeros (list n k) :dtype :float64)))
          (loop for j fixnum from 0 below k do
            (let ((sc (vt-ref s-pinv j)))
              (loop for i fixnum from 0 below n do
                (setf (vt-ref v-scaled i j) (* (vt-ref vt-mat j i) sc)))))
          (vt-@ v-scaled (vt-transpose u)))))))

;;; ============================================================
;;; 4. 最小二乘
;;; ============================================================

(defun vt-lstsq (a b &key (rcond 1e-15))
  "最小二乘解 min ||Ax - b||_2。"
  (assert (= 2 (vt-order a)))
  (with-float-safe
    (let* ((m (first (vt-shape a)))
           (n (second (vt-shape a)))
           (b-vt (ensure-vt b))
           (b-shape (vt-shape b-vt))
           (nrhs (if (= (length b-shape) 1) 1 (second b-shape)))
           (b-mat (if (= (length b-shape) 1) (vt-reshape b-vt (list m 1)) b-vt)))
      (multiple-value-bind (u s vt-mat) (vt-svd a)
        (let* ((s-data (vt-data s))
               (k (vt-size s))
               (s-max (loop for i fixnum from 0 below k maximize (aref s-data i)))
               (cutoff (* rcond s-max))
               (rank (loop for i fixnum from 0 below k count (> (aref s-data i) cutoff)))
               (utb (vt-@ (vt-transpose u) b-mat))
               (x (vt-zeros (list k nrhs) :dtype :float64)))
          (loop for j fixnum from 0 below nrhs do
            (loop for i fixnum from 0 below rank do
              (setf (vt-ref x i j) (* (vt-ref utb i j) (/ 1.0d0 (aref s-data i))))))
          (setf x (vt-@ (vt-transpose vt-mat) x))
          (let ((res (if (and (> m n) (= rank n))
                         (vt-ref (vt-norm (vt-flatten (vt-- b-mat (vt-@ a x)))))
                         0.0d0))
                (xr (if (= nrhs 1) (vt-flatten (vt-slice x '(:all) '(0))) x)))
            (values xr res rank s)))))))
