;;;; setops.lisp — 集合操作

(in-package :clvt)

(defun vt-unique (tensor &key return-index return-inverse return-counts)
  "返回展平后唯一元素（升序，nan 视为同一值）。"
  (with-float-safe
    (let* ((flat (vt-flatten tensor)) (n (vt-size flat))
           (src-data (vt-data flat)) (dtype (vt-dtype flat))
           (elem-type (vt-element-type flat))
           (sorted-idx (make-array n :element-type '(signed-byte 64)
                                     :initial-contents (loop for i below n collect i))))
      (setf sorted-idx (sort sorted-idx
                             (lambda (a b)
                               (let ((va (aref src-data a)) (vb (aref src-data b)))
                                 (cond ((vt-float-nan-p va) nil) ((vt-float-nan-p vb) t)
                                       (t (< va vb)))))))
      (let ((unique-vals (make-array 0 :element-type elem-type :adjustable t :fill-pointer t))
            (first-idx (make-array 0 :element-type '(signed-byte 64) :adjustable t :fill-pointer t))
            (cnts (make-array 0 :element-type '(signed-byte 64) :adjustable t :fill-pointer t))
            (inverse (make-array n :element-type '(signed-byte 64) :initial-element 0)))
        (loop with pos = 0
              while (< pos n)
              for uniq-num from 0
              for idx0 = (aref sorted-idx pos)
              for val = (aref src-data idx0)
              for start = pos
              do (vector-push-extend val unique-vals)
                 (vector-push-extend idx0 first-idx)
                 (if (vt-float-nan-p val)
                     (loop while (and (< pos n) (vt-float-nan-p (aref src-data (aref sorted-idx pos))))
                           for orig = (aref sorted-idx pos)
                           do (setf (aref inverse orig) uniq-num) (incf pos))
                     (loop while (and (< pos n) (= val (aref src-data (aref sorted-idx pos))))
                           for orig = (aref sorted-idx pos)
                           do (setf (aref inverse orig) uniq-num) (incf pos)))
                 (vector-push-extend (- pos start) cnts))
        (let ((uniq-vt (vt-from-sequence unique-vals :dtype dtype))
              (idx-vt (when return-index (vt-from-sequence first-idx :dtype :int64)))
              (inv-vt (when return-inverse
                        (let ((v (make-array n :element-type '(signed-byte 64))))
                          (dotimes (i n) (setf (aref v i) (aref inverse i)))
                          (%make-vt :data v :shape (list n) :strides '(1) :offset 0 :dtype :int64))))
              (cnt-vt (when return-counts (vt-from-sequence cnts :dtype :int64))))
          (if (or return-index return-inverse return-counts)
              (values uniq-vt (when return-index idx-vt) (when return-inverse inv-vt)
                      (when return-counts cnt-vt))
              uniq-vt))))))

(defun vt-intersect1d (t1 t2)
  (let* ((u1 (vt-unique t1)) (u2 (vt-unique t2))
         (u2-set (coerce (vt-data u2) 'list)) (result '()))
    (vt-do-each (ptr val u1)
      (declare (ignore ptr))
      (when (member val u2-set) (push val result)))
    (vt-from-sequence (vt-numpy-sort result #'<) :dtype (vt-dtype t1))))

(defun vt-union1d (t1 t2)
  (vt-unique (vt-concatenate 0 (vt-unique t1) (vt-unique t2))))

(defun vt-setdiff1d (t1 t2)
  (let* ((u1 (vt-unique t1)) (u2 (vt-unique t2))
         (u2-set (coerce (vt-data u2) 'list)) (result '()))
    (vt-do-each (ptr val u1)
      (declare (ignore ptr))
      (unless (member val u2-set) (push val result)))
    (vt-from-sequence (vt-numpy-sort result #'<) :dtype (vt-dtype t1))))

(defun vt-setxor1d (t1 t2)
  (let* ((u1 (vt-unique t1)) (u2 (vt-unique t2))
         (u1-set (coerce (vt-data u1) 'list)) (u2-set (coerce (vt-data u2) 'list))
         (result '()))
    (vt-do-each (ptr val u1) (declare (ignore ptr)) (unless (member val u2-set) (push val result)))
    (vt-do-each (ptr val u2) (declare (ignore ptr)) (unless (member val u1-set) (push val result)))
    (vt-from-sequence (vt-numpy-sort result #'<) :dtype (vt-dtype t1))))

(defun vt-in1d (t1 t2)
  (let ((t2-set (coerce (vt-data (vt-unique t2)) 'list)))
    (vt-map (lambda (x) (if (member x t2-set) 1.0d0 0.0d0)) t1 :dtype :float64)))
