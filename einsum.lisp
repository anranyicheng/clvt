
(in-package :clvt)

(defvar *einsum-parse-cache* (make-hash-table :test 'equal))

(declaim (inline get-parsed-subscripts))
(defun get-parsed-subscripts (str)
  (let ((cached (gethash str *einsum-parse-cache*)))
    (if cached
        (values (first cached) (second cached) (third cached))
        (multiple-value-bind (inputs output explicit-p)
            (parse-subscript-tokens-uncached str)
          (setf (gethash str *einsum-parse-cache*)
                (list inputs output explicit-p))
          (values inputs output explicit-p)))))

(defun parse-subscript-tokens-uncached (str)
  (let ((len (length str))
        (inputs nil) (current-sub nil) (output nil) (i 0) (state :inputs))
    (flet ((save-current-sub ()
             (when current-sub
               (let ((final-sub (nreverse current-sub)))
                 (if (eq state :inputs)
                     (push final-sub inputs)
                     (setf output final-sub)))
               (setf current-sub nil))))
      (loop while (< i len) do
        (let ((char (char str i)))
          (cond
            ((and (char= char #\.)
		  (< (+ i 2) len)
                  (char= (char str (+ i 1)) #\.)
		  (char= (char str (+ i 2)) #\.))
             (push :ellipsis current-sub)
	     (incf i 3))
            ((and (char= char #\-)
		  (< (1+ i) len)
		  (char= (char str (1+ i)) #\>))
             (save-current-sub)
	     (setf state :outputs) (incf i 2))
            ((char= char #\,)
	     (when (eq state :inputs)
	       (save-current-sub))
	     (incf i))
            ((char= char #\Space)
	     (incf i))
            ((char= char #\.)
	     (error "Invalid syntax: single '.' found."))
            (t (push char current-sub) (incf i)))))
      (save-current-sub)
      (values (nreverse inputs) output (if (eq state :outputs) t nil)))))

(declaim (inline expand-ellipsis))
(defun expand-ellipsis (input-subs output-sub vts)
  (let ((ellipsis-ranks nil))
    (loop for sub in input-subs
          for vt in vts
          for explicit-rank = (count-if #'characterp sub)
          for has-ellipsis = (member :ellipsis sub)
          for tensor-rank = (length (vt-shape vt))
          for implicit-rank = (if has-ellipsis
				  (- tensor-rank explicit-rank) 0)
          do (when (and has-ellipsis (< implicit-rank 0))
               (error "Subscript dimension mismatch"))
             (when (> (count :ellipsis sub) 1)
               (error "Only one ellipsis allowed"))
             (push implicit-rank ellipsis-ranks))
    (setf ellipsis-ranks (nreverse ellipsis-ranks))
    (let* ((max-implicit-rank
	     (reduce #'max ellipsis-ranks :initial-value 0))
           (ellipsis-labels
	     (loop for i from 0 below max-implicit-rank
                   collect (code-char (+ (char-code #\?) i)))))
      (flet ((expand-sub (sub implicit-rank)
               (let ((pos (position :ellipsis sub)))
                 (if (not pos) sub
                     (let* ((before (subseq sub 0 pos))
                            (after (nthcdr (1+ pos) sub))
                            (labels-to-use
			      (subseq ellipsis-labels
				      (- max-implicit-rank implicit-rank))))
                       (nconc before labels-to-use after))))))
        (values (mapcar #'expand-sub input-subs ellipsis-ranks)
                (when output-sub
		  (expand-sub output-sub max-implicit-rank)))))))

(declaim (inline analyze-einsum))
(defun analyze-einsum (input-subs output-subs vts)
  (declare (type list input-subs vts))
  (let ((label-dims (make-array 256 :element-type 'fixnum
				    :initial-element 0))
        (label-counts (make-array 256 :element-type 'fixnum
				      :initial-element 0))
        (all-labels-list nil))
    
    (loop for sub in input-subs
          for tns in vts
          for shape = (vt-shape tns)
          do (unless (= (length sub) (length shape))
               (error "Subscript dimension mismatch"))
             (loop for label in sub
                   for dim in shape
                   for code = (char-code label) do
                   (incf (aref label-counts code))
                   (pushnew label all-labels-list)
                   (let ((old-dim (aref label-dims code)))
                     (when (and (> old-dim 1) (> dim 1) (/= old-dim dim))
                       (error "Dimension conflict for label ~A: ~A vs ~A"
			      label old-dim dim))
                     (setf (aref label-dims code) (max old-dim dim)))))
    
    (let* ((final-output-subs output-subs)
           (explicit-mode (if final-output-subs t nil)))
      (unless explicit-mode
        (setf final-output-subs
              (sort (loop for code from 0 below 256
                          when (and (= (aref label-counts code) 1) 
                                    (find (code-char code) all-labels-list))
                          collect (code-char code))
                    #'char<)))
      
      (let* ((sum-labels (set-difference all-labels-list final-output-subs))
             (all-labels (coerce (append final-output-subs sum-labels) 
                                 '(simple-array character (*)))))
        (values all-labels label-dims final-output-subs)))))

(declaim (inline einsum-execute))
(defun einsum-execute
    (all-labels-vec label-dims-vec output-subs input-subs vts)
  (declare (type (simple-array character (*)) all-labels-vec)
           (type (simple-array fixnum (*)) label-dims-vec)
           (type list output-subs input-subs vts))
  
  (let* ((rank (length all-labels-vec))
         (n-vts (length vts))
         (dims-vec (make-array rank :element-type 'fixnum))
         (_ (loop for i from 0 below rank 
                  for label across all-labels-vec 
                  do (setf (aref dims-vec i)
			   (aref label-dims-vec (char-code label)))))
         
         (strides-mat (make-array (list n-vts rank) :element-type 'fixnum))
         (in-data-vec (make-array n-vts))
         (in-offsets-vec (make-array n-vts :element-type 'fixnum)))
    (declare (ignorable _))
    ;; 步长映射构建
    (loop for sub in input-subs
          for tns in vts
          for t-idx from 0 do
      (let ((phys-strides (vt-strides tns))
            (phys-shape (vt-shape tns)))
        (setf (aref in-data-vec t-idx) (vt-data tns)
              (aref in-offsets-vec t-idx) (vt-offset tns))
        (loop for lbl across all-labels-vec
              for logical-idx from 0 do
          (let ((phys-pos (position lbl sub)))
            (if phys-pos
                (let ((p-dim (nth phys-pos phys-shape))
                      (p-stride (nth phys-pos phys-strides))
                      (l-dim (aref dims-vec logical-idx)))
                  (setf (aref strides-mat t-idx logical-idx)
                        (if (and (= p-dim 1) (> l-dim 1)) 0 p-stride)))
                (setf (aref strides-mat t-idx logical-idx) 0))))))
    
    ;; 输出张量构建
    (let* ((out-shape (loop for lbl across all-labels-vec 
                            when (member lbl output-subs) 
                            collect (aref label-dims-vec (char-code lbl))))
           (output (vt-zeros out-shape))
           (out-data (vt-data output))
           (out-strides-vec (make-array rank :element-type 'fixnum)))
      
      ;; 计算输出步长
      (let ((acc 1))
        (loop for i from (1- rank) downto 0
              for dim = (aref dims-vec i) do
          (if (member (aref all-labels-vec i) output-subs)
              (progn (setf (aref out-strides-vec i) acc)
                     (setf acc (the fixnum (* acc dim))))
              (setf (aref out-strides-vec i) 0))))
      
      (let ((all-double-float-p
	      (every #'(lambda (vt)
			 (eq (vt-element-type vt) 'double-float))
		     vts)))
        (declare (type boolean all-double-float-p))
        
        (cond
          ((and (= n-vts 2)
		(= rank 3)
		(= (length output-subs) 2)
		all-double-float-p)
           (let* ((lbl-i (first output-subs))
                  (lbl-k (second output-subs))
                  (lbl-j (car (set-difference
			       (coerce all-labels-vec 'list) output-subs)))
                  (pos-i (position lbl-i all-labels-vec))
                  (pos-k (position lbl-k all-labels-vec))
                  (pos-j (position lbl-j all-labels-vec)))
             
             (when (and pos-i pos-k pos-j)
               (let* ((data-a (the (simple-array double-float (*))
				   (aref in-data-vec 0)))
                      (data-b (the (simple-array double-float (*))
				   (aref in-data-vec 1)))
                      (data-c (the (simple-array double-float (*))
				   out-data))
                      
                      (d-i (aref dims-vec pos-i))
                      (d-j (aref dims-vec pos-j))
                      (d-k (aref dims-vec pos-k))
                      
                      (sA-i (aref strides-mat 0 pos-i))
                      (sA-j (aref strides-mat 0 pos-j))
                      (sB-j (aref strides-mat 1 pos-j))
                      (sB-k (aref strides-mat 1 pos-k))
                      (sO-i (aref out-strides-vec pos-i))
                      (sO-k (aref out-strides-vec pos-k))
                      
                      (off-A (aref in-offsets-vec 0))
                      (off-B (aref in-offsets-vec 1)))
                 
                 (declare
		  (type fixnum d-i d-j d-k sA-i sA-j sB-j sB-k sO-i
			sO-k off-A off-B))
                 (declare
		  (type (simple-array double-float (*))
			data-a data-b data-c))
                 
                 (loop for i-fixnum from 0 below d-i do
                   (let ((ptr-a-row-start (+ off-A (* i-fixnum sA-i)))
                         (ptr-c-row-start (* i-fixnum sO-i)))
                     (declare (type fixnum ptr-a-row-start ptr-c-row-start))
                     (loop for j-fixnum from 0 below d-j do
                       (let ((val-a (aref data-a (+ ptr-a-row-start
						    (* j-fixnum sA-j)))))
                         (declare (type double-float val-a))
                         (let ((ptr-b-start (+ off-B (* j-fixnum sB-j)))
                               (ptr-c-start ptr-c-row-start))
                           (declare (type fixnum ptr-b-start ptr-c-start))
                           (loop for k-fixnum from 0 below d-k do
                             (incf (aref data-c ptr-c-start) 
                                   (* val-a (aref data-b ptr-b-start)))
                             (incf ptr-b-start sB-k)
                             (incf ptr-c-start sO-k)))))))
                 (return-from einsum-execute output)))))
          (t  ;; 通用路径
           (let ((cur-ptrs (make-array n-vts :element-type 'fixnum
					     :initial-element 0)))
             (loop for k from 0 below n-vts do
	       (setf (aref cur-ptrs k) (aref in-offsets-vec k)))
             (labels
		 ((recurse (depth out-ptr)
                    (declare (type fixnum depth out-ptr))
                    (if (= depth rank)
                        (let ((product 1.0d0))
                          (declare (type double-float product))
                          (loop for k fixnum from 0 below n-vts do
                            (setf product (* product 
                                             (aref (aref in-data-vec k)
                                                   (aref cur-ptrs k)))))
                          (incf (aref out-data out-ptr) product))
                        
                        (let* ((dim (aref dims-vec depth))
                               (out-stride (aref out-strides-vec depth)))
                          (declare (type fixnum dim out-stride))
                          (loop for i fixnum from 0 below dim do
                            (recurse (1+ depth) out-ptr)
                            ;; 推进指针
                            (loop for k fixnum from 0 below n-vts do
                              (incf (aref cur-ptrs k)
				    (aref strides-mat k depth)))
                            (incf out-ptr out-stride))
                          (loop for k fixnum from 0 below n-vts do
                            (decf (aref cur-ptrs k)
				  (the fixnum
				       (* dim (aref strides-mat k depth)))))
                          (decf out-ptr (the fixnum (* dim out-stride)))))))
               
               (recurse 0 0)))))
        output))))

(defun vt-einsum (subscripts &rest vts)
  "高性能 Einsum 接口."
  (multiple-value-bind (raw-inputs raw-output explicit-p)
      (get-parsed-subscripts subscripts)
    (unless (= (length raw-inputs) (length vts))
      (error "提供的张量数量 ~A 与子脚标 ~A 不匹配" (length vts) raw-inputs))
    
    (multiple-value-bind (input-subs output-subs)
        (expand-ellipsis raw-inputs raw-output vts)
      (unless explicit-p (setf output-subs nil))
      (multiple-value-bind (all-labels label-dims output-subs-final)
          (analyze-einsum input-subs output-subs vts)
        (einsum-execute
	 all-labels label-dims output-subs-final input-subs vts)))))
