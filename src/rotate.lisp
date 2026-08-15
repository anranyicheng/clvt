(in-package :clvt)


(defparameter *clvt-pi180* 1.74532925199432957692d-2)

(defparameter *clvt-sincof*
  #(1.58962301572218447952d-10 -2.50507477628503540135d-8
    2.75573136213856773549d-6 -1.98412698295895384658d-4
    8.33333333332211858862d-3 -1.66666666666666307295d-1))

(defparameter *clvt-coscof*
  #(1.13678171382044553091d-11 -2.08758833757683644217d-9
    2.75573155429816611547d-7 -2.48015872936186303776d-5
    1.38888888888806666760d-3 -4.16666666666666348141d-2
    4.99999999999999999798d-1))

(defun cephes-polevl (x coef n)
  "cephes horner 多项式求值: coef[0]*x^n + ... + coef[n]"
  (let ((ans (aref coef 0)))
    (do ((i 1 (1+ i)))
        ((> i n) ans)
      (declare (type fixnum i))
      (setf ans (+ (* ans x) (aref coef i))))))

(defun cephes-sindg (x)
  "完全复刻 c 源码 sindg"
  (declare (type double-float x))
  (let ((sign 1d0)
        (xx x))
    (declare (type double-float sign xx))
    (when (< xx 0d0)
      (setf sign -1d0)
      (setf xx (- xx)))
    
    (let* ((y (float (floor (/ xx 45d0)) 1.0d0))
           (z (float (floor (/ y 16d0)) 1.0d0))
           (z (- y (* z 16d0)))
           (j (truncate z)))
      (declare (type double-float y z) (type fixnum j))
      (when (= 1 (logand j 1))
        (incf j)
        (incf y))
      (setf j (logand j 7))
      (when (> j 3)
        (setf sign (- sign))
        (decf j 4))
      (let* ((x2 (* (- xx (* y 45d0)) *clvt-pi180*))
             (zz (* x2 x2))
             (res (if (or (= j 1) (= j 2))
                      (- 1d0 (* zz (cephes-polevl zz *clvt-coscof* 6)))
                      (+ x2 (* x2 (* zz (cephes-polevl zz *clvt-sincof* 5)))))))
        (declare (type double-float res))
        (if (< sign 0) (- res) res)))))

(defun cephes-cosdg (x)
  "完全复刻 c 源码 cosdg"
  (declare (type double-float x))
  (let* ((sign 1d0)
         (xx (if (< x 0d0) (- x) x)))
    (declare (type double-float sign xx))
    (let* ((y (float (floor (/ xx 45d0)) 1.0d0))
           (z (float (floor (/ y 16d0)) 1.0d0))
           (z (- y (* z 16d0)))
           (j (truncate z)))
      (declare (type double-float y z) (type fixnum j))
      (when (= 1 (logand j 1))
        (incf j)
        (incf y))
      (setf j (logand j 7))
      (when (> j 3)
        (setf sign (- sign))
        (decf j 4))
      (when (> j 1)
        (setf sign (- sign)))
      (let* ((x2 (* (- xx (* y 45d0)) *clvt-pi180*))
             (zz (* x2 x2))
             (res (if (or (= j 1) (= j 2))
                      (+ x2 (* x2 (* zz (cephes-polevl zz *clvt-sincof* 5))))
                      (- 1d0 (* zz (cephes-polevl zz *clvt-coscof* 6))))))
        (declare (type double-float res))
        (if (< sign 0) (- res) res)))))

;;  map_coordinate (1:1 复刻 c 源码边界逻辑)
(defun map-coordinate (in len mode)
  (declare (type double-float in) (type fixnum len))
  (cond
    ((< in 0.0d0)
     (case mode
       (:mirror
        (if (<= len 1) 0.0d0
            (let* ((sz2 (- (* 2 len) 2))  ;; 2*len - 2
					  (n (truncate (/ (- in) sz2))))
              (declare (type fixnum sz2 n))
              (setq in (+ (* sz2 n) in))
              (if (<= in (- 1 len)) (+ in sz2) (- in)))))
       (:reflect
        (if (<= len 1) 0.0d0
            (let ((sz2 (* 2 len)))
              (declare (type fixnum sz2))
              (when (< in (- sz2))
                (setq in (+ (* sz2 (truncate (/ (- in) sz2))) in)))
              (if (< in (- len))
                  (+ in sz2)
                  (- (if (> in -1d-15) 1d-15 (- in)) 1)))))
       (:wrap
        (if (<= len 1) 0.0d0
            (let* ((sz (- len 1)))  ;; len - 1
              (declare (type fixnum sz))
              (+ in (* sz (+ (truncate (/ (- in) sz)) 1))))))
       (:grid-wrap
        (if (<= len 1) 0.0d0
            (+ in (* len (+ (truncate (/ (- -1 in) len)) 1)))))
       (:nearest 0.0d0)
       (t -1.0d0)))
    ((> in (coerce (- len 1) 'double-float))
     (case mode
       (:mirror
        (if (<= len 1) 0.0d0
            (let* ((sz2 (- (* 2 len) 2)))  ;; 2*len - 2
              (declare (type fixnum sz2))
              (setq in (- in (* sz2 (truncate (/ in sz2)))))
              (if (>= in len) (- sz2 in) in))))
       (:reflect
        (if (<= len 1) 0.0d0
            (let ((sz2 (* 2 len)))
              (declare (type fixnum sz2))
              (setq in (- in (* sz2 (truncate (/ in sz2)))))
              (if (>= in len) (- sz2 in 1) in))))
       (:wrap
        (if (<= len 1) 0.0d0
            (let* ((sz (- len 1)))  ;; len - 1
              (declare (type fixnum sz))
              (- in (* sz (truncate (/ in sz)))))))
       (:grid-wrap
        (if (<= len 1) 0.0d0
            (- in (* len (truncate (/ in len))))))
       (:nearest (coerce (- len 1) 'double-float))
       (t -1.0d0)))
    (t in)))


;;  形状计算与主旋转函数
(defun compute-rotated-shape (rows cols cos-a sin-a)
  "复刻 python: out_bounds = rot_matrix @ corners; shape = floor(ptp + 0.5)"
  (let* ((iy rows) (ix cols)
		   (ob00 0.0d0)
		   (ob01 (* sin-a (coerce ix 'double-float)))
		   (ob02 (* cos-a (coerce iy 'double-float)))
		   (ob03 (+ (* cos-a (coerce iy 'double-float))
			    (* sin-a (coerce ix 'double-float))))
		   (ob10 0.0d0)
		   (ob11 (* cos-a (coerce ix 'double-float)))
		   (ob12 (* (- sin-a) (coerce iy 'double-float)))
		   (ob13 (+ (* (- sin-a) (coerce iy 'double-float))
			    (* cos-a (coerce ix 'double-float))))
		   (ptp0 (- (max ob00 ob01 ob02 ob03) (min ob00 ob01 ob02 ob03)))
		   (ptp1 (- (max ob10 ob11 ob12 ob13) (min ob10 ob11 ob12 ob13))))
    (values (max 1 (floor (+ ptp0 0.5d0)))
            (max 1 (floor (+ ptp1 0.5d0))))))

(defun vt-rotate (tensor angle &key
				 (reshape nil)
				 (order 1)
				 (mode :constant)
				 (cval 0.0)
				 (center nil))
  "将二维张量按指定角度旋转（完全对标 scipy.ndimage.rotate 的行为与精度）。
   
   **角度单位与方向**：
   - angle 的单位为**度（degrees）**，而非弧度。
   - angle 为正时表示逆时针旋转，为负时表示顺时针旋转。
   - 底层调用精准复刻的 cephes sindg/cosdg 算法，确保与 scipy 浮点结果按位级一致。

   **旋转中心**：
   - 默认 center = nil，旋转中心位于张量的几何中心，即坐标 ((rows-1)/2, (cols-1)/2)。
   - 若提供 center 列表 (cy, cx)，则绕该指定点旋转。例如 '(0 0) 表示绕左上角原点旋转。
   - 注意：行坐标对应 cy，列坐标对应 cx，与数组索引顺序一致（行在前，列在后）。

   **画布大小**：
   - reshape = nil (默认)：保持原始张量的形状不变，旋转后超出边界的部分会被截断。
   - reshape = t：自动扩大输出张量的画布（形状重新计算），以完整包含旋转后的所有内容。

   **插值方式**：
   - order = 0 : 最近邻插值（直接取整，速度快，边缘有锯齿，像素值不改变）。
   - order = 1 : 双线性插值（平滑过渡，但计算量稍大，会产生新的像素值）。
   - 注意：目前仅支持 order 0 和 1，底层采用与 scipy 完全一致的样条权重计算逻辑。

   **边界处理**：
   - 当目标像素映射到原始张量边界之外时，根据 mode 参数处理：
     - :constant (默认): 使用 cval 填充边界（默认 0.0）。
     - :nearest, :mirror, :reflect, :wrap, :grid-wrap, :grid-constant: 
       提供与 scipy.ndimage 完全一致的边界反射与平铺模式。
   - 内部通过逆向映射实现：从目标张量的每个像素出发反推原始坐标进行采样，保证旋转后图像无空洞。

   **返回值**：
   - 返回一个新的二维张量（非视图），其元素数据类型与输入 tensor 保持一致。

   **示例**：
   ;; 绕图像中心逆时针旋转 45°，双线性插值，保持原画布大小
   (let ((m (clvt:vt-from-sequence '((1 2 3 4) (5 6 7 8) (9 10 11 12) (13 14 15 16)))))
     (vt-rotate m 45 :order 1 :mode :constant))

   ;; 绕左上角原点逆时针旋转 30°，最近邻插值，并自动扩展画布
   (let ((m (clvt:vt-from-sequence '((1 0) (0 1)))))
     (vt-rotate m 30 :order 0 :reshape t :center '(0 0)))
  "
  (let* ((shape (vt-shape tensor))
         (rows (first shape))
         (cols (second shape))
         (angle-d (coerce angle 'double-float))
         ;; 使用精准复刻的 cephes cosdg/sindg
         (cos-a (cephes-cosdg angle-d))
         (sin-a (cephes-sindg angle-d))
         (in-data (vt-data tensor))
         (s0 (first (vt-strides tensor)))
         (s1 (second (vt-strides tensor)))
         (off (vt-offset tensor))
         (cval-d (coerce cval 'double-float))
         (out-type (vt-dtype tensor)))
    (multiple-value-bind (out-rows out-cols)
        (if reshape
            (compute-rotated-shape rows cols cos-a sin-a)
            (values rows cols))
      (let* ((out-cy (if center (first center) (/ (- out-rows 1) 2.0d0)))
             (out-cx (if center (second center) (/ (- out-cols 1) 2.0d0)))
             (in-cy (if center (first center) (/ (- rows 1) 2.0d0)))
             (in-cx (if center (second center) (/ (- cols 1) 2.0d0)))
             ;; offset = in_center - rot_matrix @ out_center
             (off-y (- in-cy (+ (* out-cy cos-a) (* out-cx sin-a))))
             (off-x (- in-cx (+ (* out-cy (- sin-a)) (* out-cx cos-a))))
             (out-vt (vt-zeros (list out-rows out-cols) :dtype out-type))
             (out-data (vt-data out-vt)))
        (loop for ny fixnum from 0 below out-rows
              do (loop for nx fixnum from 0 below out-cols
                       do (let* (;; 关键：复刻 c 代码的运算顺序
                                 ;; tmp = shift; tmp += ny*c; tmp += nx*s;
                                 (src-y (+ off-y (* ny cos-a) (* nx sin-a)))
                                 (src-x (+ off-x (* ny (- sin-a))
					   (* nx cos-a))))
                            (setf (aref out-data (+ (* ny out-cols) nx))
                                  (vt-cast
				   (sample-pixel src-x src-y rows cols
                                                 in-data s0 s1 off
                                                 order mode cval-d)
                                   out-type)))))
        out-vt))))

;;  像素采样函数 (1:1 复刻 c 源码 ni_geometrictransform)

(defun get-spline-boundary-mode (mode)
  "c 源码中，constant 和 wrap 的样条边界模式会被强制改为 mirror"
  (if (or (eq mode :constant) (eq mode :wrap))
      :mirror
      mode))

(defun sample-pixel (src-x src-y rows cols in-data s0 s1 off order mode cval)
  (declare (type double-float src-x src-y cval)
           (type fixnum rows cols s0 s1 off order))
  (let ((spline-mode (get-spline-boundary-mode mode))
        (cc-y src-y) (cc-x src-x))
    
    ;; 1. 仅当非 grid-constant 和非 nearest 时调用 map_coordinate
    (unless (or (eq mode :grid-constant) (eq mode :nearest))
      (setq cc-y (map-coordinate src-y rows mode))
      (setq cc-x (map-coordinate src-x cols mode)))
    
    ;; 2. 检查是否触发 constant 边界 (c: if cc > -1.0 || grid_const || nearest)
    (if (or (eq mode :grid-constant) (eq mode :nearest)
            (and (> cc-y -1.0d0) (> cc-x -1.0d0)))
        ;; 3. 计算 start 索引
        (let ((start-y (if (= (logand order 1) 1)
                           (- (floor cc-y) (floor order 2))
                           (- (floor (+ cc-y 0.5d0)) (floor order 2))))
              (start-x (if (= (logand order 1) 1)
                           (- (floor cc-x) (floor order 2))
                           (- (floor (+ cc-x 0.5d0)) (floor order 2)))))
          (declare (type fixnum start-y start-x))
          
          ;; 检查 edge
          (let ((edge-y (or (< start-y 0) (>= (+ start-y order) rows)))
                (edge-x (or (< start-x 0) (>= (+ start-x order) cols))))
            (if (= order 0)
                ;; order=0: 最近邻
                (let ((iy start-y) (ix start-x))
                  (declare (type fixnum iy ix))
                  (when edge-y
                    (setq iy (truncate (map-coordinate
					(coerce start-y 'double-float)
					rows spline-mode))))
                  (when edge-x
                    (setq ix (truncate (map-coordinate
					(coerce start-x 'double-float)
					cols spline-mode))))
                  (aref in-data (+ off (* iy s0) (* ix s1))))
                ;; order=1: 双线性
                (let* (;; spline weights: fc = cc - floor(cc)
                       (fy (- cc-y (coerce (floor cc-y) 'double-float)))
                       (fx (- cc-x (coerce (floor cc-x) 'double-float)))
                       (fy0 (- 1.0d0 fy))
                       (fx0 (- 1.0d0 fx))
                       (y0 start-y) (y1 (1+ start-y))
                       (x0 start-x) (x1 (1+ start-x)))
                  (declare (type double-float fy fx fy0 fx0)
                           (type fixnum y0 y1 x0 x1))
                  ;; edge 映射使用 spline_mode
                  (when edge-y
                    (setq y0 (truncate (map-coordinate
					(coerce y0 'double-float)
					rows spline-mode)))
                    (setq y1 (truncate (map-coordinate
					(coerce y1 'double-float)
					rows spline-mode))))
                  (when edge-x
                    (setq x0 (truncate (map-coordinate
					(coerce x0 'double-float)
					cols spline-mode)))
                    (setq x1 (truncate (map-coordinate
					(coerce x1 'double-float)
					cols spline-mode))))
                  (let ((p00 (aref in-data (+ off (* y0 s0) (* x0 s1))))
                        (p10 (aref in-data (+ off (* y0 s0) (* x1 s1))))
                        (p01 (aref in-data (+ off (* y1 s0) (* x0 s1))))
                        (p11 (aref in-data (+ off (* y1 s0) (* x1 s1)))))
                    ;; 关键：复刻 c 代码的乘法和加法顺序
                    ;; coeff = pixel; coeff *= splvals[0]; coeff *= splvals[1];
                    ;; t += coeff;
                    (+ (* (* p00 fy0) fx0)
                       (* (* p10 fy0) fx)
                       (* (* p01 fy) fx0)
                       (* (* p11 fy) fx)))))))
        ;; 触发 cval
        cval)))

(defun vt-rotate-origin
    (tensor angle &key (order 0) (reshape nil) (mode :constant) (cval 0.0))
  "绕左上角原点 (0,0) 旋转张量，是 vt-rotate 的便捷版本。
       等价于 (vt-rotate tensor angle :center '(0 0) :order order ...)。
       该函数将张量的 (0,0) 点视为旋转中心，逆时针旋转 angle 度（注意是度数，不是弧度）。
       如果 order=0（默认），像素值仅位置发生变化，不改变数值；
       如果 order=1，则进行双线性插值，像素值会发生变化。
       超出原张量边界的部分用 cval 填充（默认为 0）。"
  (vt-rotate tensor angle 
             :center (list 0.0d0 0.0d0) 
             :order order 
             :reshape reshape 
             :mode mode 
             :cval cval))
