(in-package :clvt)

;;; 随机数扩展
(defvar *vt-default-random-state* (make-random-state *random-state*)
  "clvt 内部使用的默认随机状态。可由 vt-random-seed 修改或通过 let 覆盖。")

(defun vt-make-random-state (&optional seed)
  "创建并返回一个新的 random-state 对象。seed 同 make-random-state 参数。
   示例：
     (vt-make-random-state)     => 使用 t 随机初始化
     (vt-make-random-state nil) => 从当前 *random-state* 复制
     (vt-make-random-state 42)  => 不可移植地使用整数初始化"
  #+sbcl
  (sb-ext::seed-random-state seed)
  #-sbcl
  (typecase seed
    (null (make-random-state nil)) ; 保持与 cl 一致：nil → 复制当前全局状态
    (random-state (make-random-state seed))
    (t (make-random-state seed)))  ; 实现依赖，通常可为整数加载
  )

(defun vt-random-seed (seed)
  "修改 *vt-default-random-state* 并返回新的状态。seed 含义同 vt-make-random-state。
   例：(vt-random-seed 42)  后，所有使用默认 rng 的函数将产生可复现序列。"
  (setf *vt-default-random-state* (vt-make-random-state seed)))

(declaim (inline %uniform-rand %normal-rand))

(defun %uniform-rand (state)
  "生成 [0,1) 均匀分布随机数。"
  (declare (random-state state))
  (random 1.0d0 state))

(defun %normal-rand (state)
  "box-muller 方法生成标准正态分布随机数。"
  (declare (random-state state))
  (let ((u1 (max least-positive-double-float (random 1.0d0 state)))
        (u2 (random 1.0d0 state)))
    (* (sqrt (* -2.0d0 (log u1)))
       (cos (* 2.0d0 pi u2)))))

(defun vt-random (shape &key (dtype :float64)
			  (rng *vt-default-random-state*))
  "返回形状为 shape 的张量，元素独立同分布于 [0,1) 均匀分布。"
  (declare (list shape) (random-state rng))
  (vt-map (lambda (x)
            (declare (ignore x))
            (vt-cast (%uniform-rand rng) dtype))
          (vt-zeros shape :dtype dtype)))

(defun vt-random-uniform
    (shape &key (low 0.0d0) (high 1.0d0) (dtype :float64)
	     (rng *vt-default-random-state*))
  "返回 [low, high) 均匀分布随机张量。"
  (declare (list shape) (random-state rng))
  (let ((range (- high low)))
    (vt-map (lambda (x)
              (declare (ignore x))
              (vt-cast (+ low (* range (%uniform-rand rng))) dtype))
            (vt-zeros shape :dtype dtype))))

(defun vt-random-normal (shape &key (mean 0.0d0) (std 1.0d0)
				 (rng *vt-default-random-state*))
  "返回正态分布随机张量。"
  (declare (list shape) (random-state rng))
  (let ((res (vt-zeros shape :dtype :float64)))
    (vt-do-each (ptr val res)
      (declare (ignore val))
      (setf (aref (vt-data res) ptr)
            (+ mean (* std (%normal-rand rng)))))
    res))

(defun vt-random-int (low high &key (size nil) (dtype :int64)
				 (rng *vt-default-random-state*))
  "创建随机整数数组.
  low: 下界(包含)
  high: 上界(不包含)
  size: 形状(nil 表示标量)
  返回: 张量"
  (declare (random-state rng))
  (let ((range (- high low)))
    (assert (>= range 0)
	    (high low)
	    "high: ~a less than low: ~a" high low)
    (if size
        (vt-astype (vt-map (lambda (x)
                             (declare (ignore x))
                             (+ low (random range rng)))
                           (vt-zeros size :dtype dtype))
                   dtype)
	(make-vt nil (+ low (random range rng)) :dtype dtype))))

(defun vt-random-integers (low high &key (size nil) (dtype :int64)
				      (rng *vt-default-random-state*))
  "同 vt-random-int。"
  (vt-random-int low high :size size :dtype dtype :rng rng))

