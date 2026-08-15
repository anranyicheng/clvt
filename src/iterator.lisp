;;;; iterator.lisp — 迭代原语（单一张量通用遍历宏）

(in-package :clvt)

(defmacro vt-do-each ((ptr-var val-var vt) &body body)
  "遍历张量 vt 的每个元素（支持非连续视图）。
   ptr-var 绑定当前元素的物理索引，val-var 绑定其值。"
  (let ((shape (gensym "SHAPE"))
        (strides (gensym "STRIDES"))
        (data (gensym "DATA"))
        (offset (gensym "OFFSET"))
        (rank (gensym "RANK"))
        (dims (gensym "DIMS"))
        (strs (gensym "STRS"))
        (depth (gensym "DEPTH")))
    `(let* ((,shape (vt-shape ,vt))
            (,strides (vt-strides ,vt))
            (,data (vt-data ,vt))
            (,offset (vt-offset ,vt))
            (,rank (length ,shape))
            (,dims (coerce ,shape 'simple-vector))
            (,strs (coerce ,strides 'simple-vector)))
       (labels ((recurse (,depth ptr)
                  (if (= ,depth ,rank)
                      (let ((,ptr-var ptr)
                            (,val-var (aref ,data ptr)))
                        (declare (ignorable ,ptr-var ,val-var))
                        ,@body)
                      (let ((dim (svref ,dims ,depth))
                            (stride (svref ,strs ,depth)))
                        (loop for i fixnum from 0 below dim do
                          (recurse (1+ ,depth) ptr)
                          (incf ptr stride))))))
         (recurse 0 ,offset)))))
