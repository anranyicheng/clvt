;;;; vt-manip-interop-test.lisp
;;;; =============================================================================
;;;; clvt 形状/视图/拼接/切片类 API 的跨语言互操作测试 —— Lisp 端生成器
;;;; =============================================================================
;;;;
;;;; 目标（不包含数值计算部分，只考察「结构正确性」）：
;;;;   reshape / 转换 / 拼接 / 切片 / 连续化 / 索引选择 / 类型转换
;;;;
;;;; 规模基准（用户指定）：
;;;;     (vt-reshape (vt-arange 1000000) '(1000 1000))
;;;;
;;;; 工作方式：
;;;;   1. 在 SBCL + clvt 下把每个用例的结果用 `vt-save` 落盘：
;;;;        - 全部用例写 :npy  （与 np.save 逐字节兼容，供 NumPy 读取）
;;;;        - 小结果用例额外写 :lisp（原生 s-expression 格式，用于往返自检）
;;;;   2. 同时做 Lisp 端独立自检：
;;;;        - :lisp / :npy 落盘后 `vt-load` 回读，`vt-array-equal` 逐位比对
;;;;        - 一批「手算 oracle」断言（不依赖 NumPy，例如转置元素位置）
;;;;        - 一批「错误契约」断言（越界/形状不匹配必须报错）
;;;;   3. 生成 manifest.json 描述每个用例的形式、形状、dtype、输出文件，
;;;;      交由 test/verify_manip_numpy.py 用 NumPy 逐项校验。
;;;;
;;;; 运行（经统一入口，推荐）：
;;;;     bash test/run-tests.sh --suite manip-interop-test
;;;; 或单独跑本阶段（结果留在 CLVT_TEST_OUT 供 stage 2 校验）：
;;;;     CLVT_TEST_OUT=/tmp/clvt-interop-out sbcl --script test/vt-manip-interop-test.lisp
;;;; =============================================================================

(require :asdf)

;; 加载 clvt：经 run-tests.sh 调用时由运行器先行加载；独立运行（--script 不含
;; userinit，没有 quicklisp）时从仓库根目录用 ASDF 加载，不依赖 quicklisp。
(eval-when (:load-toplevel :execute)
  (unless (find-package :clvt)
    (let* ((self (or *load-truename* *default-pathname-defaults*))
           (dir (make-pathname :name nil :type nil :version nil :defaults self))
           (root (truename (merge-pathnames #p"../" dir))))
      (pushnew root asdf:*central-registry*))
    (asdf:load-system :clvt)))

(in-package :clvt)

;;; ---------------------------------------------------------------------------
;;; 0. 基础设施
;;; ---------------------------------------------------------------------------

(defparameter *out-dir*
  (or (sb-ext:posix-getenv "CLVT_TEST_OUT") "/tmp/clvt-interop-out")
  "输出目录（npy / lisp / manifest 都写到这里）。")

(defparameter *lisp-max-elems* 250000
  "结果元素数不超过该阈值的用例，额外写 :lisp 原生格式并做往返自检。")

(defun out-dir-path ()
  "把 *OUT-DIR* 规范化为目录 pathname（保证 merge-pathnames 追加而非替换）。"
  (pathname (concatenate 'string
                         (string-right-trim "/" *out-dir*)
                         "/")))

(defun out-path (name)
  "输出目录下的文件路径。"
  (merge-pathnames name (out-dir-path)))

(defvar *cases* '()
  "用例表： (name form-string thunk)。存放顺序为逆序，输出前 reverse。")

(defmacro defcase (name form &body body)
  "注册一个用例。BODY 返回单个 vt 或 vt 列表。"
  `(push (list ,name ,form (lambda () ,@body)) *cases*))

(defun normalize-results (r)
  "把用例返回值归一为 vt 列表。"
  (cond
    ((vt-p r) (list r))
    ((and (consp r) (every #'vt-p r)) r)
    (t (error "用例返回了非张量结果: ~S" r))))

(defun json-escape (s)
  (with-output-to-string (o)
    (loop for ch across s do
      (case ch
        (#\\ (write-string "\\\\" o))
        (#\" (write-string "\\\"" o))
        (#\Newline (write-string "\\n" o))
        (#\Return (write-string "\\r" o))
        (#\Tab (write-string "\\t" o))
        (t (write-char ch o))))))

(defun json-str (s) (format nil "\"~A\"" (json-escape (string s))))
(defun json-int-list (lst)
  (format nil "[~{~D~^, ~}]" lst))

(defun dtype-name (tensor)
  "clvt dtype 关键字 -> numpy 风格小写名。"
  (string-downcase (symbol-name (vt-dtype tensor))))

;;; ---------------------------------------------------------------------------
;;; 1. 基准数据（用户指定规模）
;;; ---------------------------------------------------------------------------

(defparameter *base* (vt-arange 1000000)                       ; (1000000,) float64 0..999999
  "基准一维数据：(vt-arange 1000000)，默认 dtype :float64。")
(defparameter *m* (vt-reshape *base* '(1000 1000))
  "基准矩阵：(vt-reshape (vt-arange 1000000) '(1000 1000))。")
(defparameter *m3* (vt-reshape (vt-arange 1000000) '(100 100 100))
  "基准三维张量，同源数据。")

;;; ---------------------------------------------------------------------------
;;; 2. 用例注册 —— A 组：view / reshape（转换）
;;; ---------------------------------------------------------------------------

(defcase "A01_view_same_shape"
  "(vt-view m '(1000 1000))"
  (vt-view *m* '(1000 1000)))
(defcase "A02_view_flat"
  "(vt-view m '(1000000))"
  (vt-view *m* '(1000000)))
(defcase "A03_view_500x2000"
  "(vt-view m '(500 2000))"
  (vt-view *m* '(500 2000)))
(defcase "A04_view_100x100x100"
  "(vt-view m '(100 100 100))"
  (vt-view *m* '(100 100 100)))
(defcase "A05_view_2000x500"
  "(vt-view m '(2000 500))"
  (vt-view *m* '(2000 500)))
(defcase "A06_reshape_same"
  "(vt-reshape m '(1000 1000))"
  (vt-reshape *m* '(1000 1000)))
(defcase "A07_reshape_flat"
  "(vt-reshape m '(1000000))"
  (vt-reshape *m* '(1000000)))
(defcase "A08_reshape_4d"
  "(vt-reshape m '(10 10 100 100))"
  (vt-reshape *m* '(10 10 100 100)))
(defcase "A09_reshape_transposed_flat"
  "(vt-reshape (vt-transpose m) '(1000000))"
  (vt-reshape (vt-transpose *m*) '(1000000)))
(defcase "A10_reshape_transposed_500x2000"
  "(vt-reshape (vt-transpose m) '(500 2000))"
  (vt-reshape (vt-transpose *m*) '(500 2000)))
(defcase "A11_reshape_narrowed"
  "(vt-reshape (vt-narrow m 0 100 700) '(600 1000))"
  (vt-reshape (vt-narrow *m* 0 100 700) '(600 1000)))
(defcase "A12_reshape_strided_cols"
  "(vt-reshape (vt-slice m '(:all) '(0 1000 2)) '(1000 500))"
  (vt-reshape (vt-slice *m* '(:all) '(0 1000 2)) '(1000 500)))

;;; ---------------------------------------------------------------------------
;;; 3. B 组：transpose / swapaxes / rot90 / flip / roll
;;; ---------------------------------------------------------------------------

(defcase "B01_transpose_rev"
  "(vt-transpose m)"
  (vt-transpose *m*))
(defcase "B02_transpose_perm_201"
  "(vt-transpose m3 '(2 0 1))"
  (vt-transpose *m3* '(2 0 1)))
(defcase "B03_transpose_perm_120"
  "(vt-transpose m3 '(1 2 0))"
  (vt-transpose *m3* '(1 2 0)))
(defcase "B04_transpose_3d_rev"
  "(vt-transpose m3)"
  (vt-transpose *m3*))
(defcase "B05_transpose_of_narrow"
  "(vt-transpose (vt-narrow m 0 100 700))"
  (vt-transpose (vt-narrow *m* 0 100 700)))
(defcase "B06_swapaxes_0_1"
  "(vt-swapaxes m 0 1)"
  (vt-swapaxes *m* 0 1))
(defcase "B07_swapaxes_m3_0_2"
  "(vt-swapaxes m3 0 2)"
  (vt-swapaxes *m3* 0 2))
(defcase "B08_rot90_k1"
  "(vt-rot90 m)"
  (vt-rot90 *m*))
(defcase "B09_rot90_k2"
  "(vt-rot90 m :k 2)"
  (vt-rot90 *m* :k 2))
(defcase "B10_rot90_k3"
  "(vt-rot90 m :k 3)"
  (vt-rot90 *m* :k 3))
(defcase "B11_rot90_k-1"
  "(vt-rot90 m :k -1)"
  (vt-rot90 *m* :k -1))
(defcase "B12_rot90_3d_axes12"
  "(vt-rot90 m3 :axes '(1 2))"
  (vt-rot90 *m3* :axes '(1 2)))
(defcase "B13_flip_ax0"
  "(vt-flip m :axis 0)"
  (vt-flip *m* :axis 0))
(defcase "B14_flip_ax1"
  "(vt-flip m :axis 1)"
  (vt-flip *m* :axis 1))
(defcase "B15_flip_all"
  "(vt-flip m :axis nil)"
  (vt-flip *m* :axis nil))
(defcase "B16_roll_ax0_5"
  "(vt-roll m 5 :axis 0)"
  (vt-roll *m* 5 :axis 0))
(defcase "B17_roll_ax1_-3"
  "(vt-roll m -3 :axis 1)"
  (vt-roll *m* -3 :axis 1))
(defcase "B18_roll_two_axes"
  "(vt-roll m '(5 7) :axis '(0 1))"
  (vt-roll *m* '(5 7) :axis '(0 1)))
(defcase "B19_roll_flat"
  "(vt-roll m 123)"
  (vt-roll *m* 123))
(defcase "B20_roll_flat_list"
  "(vt-roll m '(1000000 5) :axis nil)"
  (vt-roll *m* '(1000000 5) :axis nil))

;;; ---------------------------------------------------------------------------
;;; 4. C 组：squeeze / unsqueeze / expand-dims / flatten / ravel
;;; ---------------------------------------------------------------------------

(defcase "C01_squeeze_all"
  "(vt-squeeze (vt-reshape m '(1 1000 1000 1)))"
  (vt-squeeze (vt-reshape *m* '(1 1000 1000 1))))
(defcase "C02_squeeze_axis0"
  "(vt-squeeze (vt-reshape m '(1 1000 1000)) :axis 0)"
  (vt-squeeze (vt-reshape *m* '(1 1000 1000)) :axis 0))
(defcase "C03_squeeze_axis2"
  "(vt-squeeze (vt-reshape m '(1000 1000 1)) :axis 2)"
  (vt-squeeze (vt-reshape *m* '(1000 1000 1)) :axis 2))
(defcase "C04_unsqueeze0"
  "(vt-unsqueeze m 0)"
  (vt-unsqueeze *m* 0))
(defcase "C05_unsqueeze1"
  "(vt-unsqueeze m 1)"
  (vt-unsqueeze *m* 1))
(defcase "C06_unsqueeze_neg1"
  "(vt-unsqueeze m -1)"
  (vt-unsqueeze *m* -1))
(defcase "C07_expand_dims0"
  "(vt-expand-dims m 0)"
  (vt-expand-dims *m* 0))
(defcase "C08_expand_dims2"
  "(vt-expand-dims m 2)"
  (vt-expand-dims *m* 2))
(defcase "C09_flatten"
  "(vt-flatten m)"
  (vt-flatten *m*))
(defcase "C10_flatten_transposed"
  "(vt-flatten (vt-transpose m))"
  (vt-flatten (vt-transpose *m*)))
(defcase "C11_ravel"
  "(vt-ravel m)"
  (vt-ravel *m*))
(defcase "C12_ravel_transposed"
  "(vt-ravel (vt-transpose m))"
  (vt-ravel (vt-transpose *m*)))
(defcase "C13_ravel_strided_slice"
  "(vt-ravel (vt-slice m '(0 1000 3) '(:all)))"
  (vt-ravel (vt-slice *m* '(0 1000 3) '(:all))))

;;; ---------------------------------------------------------------------------
;;; 5. D 组：narrow / slice（切片）
;;; ---------------------------------------------------------------------------

(defcase "D01_narrow_rows"
  "(vt-narrow m 0 100 900)"
  (vt-narrow *m* 0 100 900))
(defcase "D02_narrow_cols"
  "(vt-narrow m 1 0 500)"
  (vt-narrow *m* 1 0 500))
(defcase "D03_narrow_neg_start"
  "(vt-narrow m 0 -100 1000)"
  (vt-narrow *m* 0 -100 1000))
(defcase "D04_narrow_3d_axis1"
  "(vt-narrow m3 1 10 90)"
  (vt-narrow *m3* 1 10 90))
(defcase "D05_slice_rows"
  "(vt-slice m '(100 900) '(:all))"
  (vt-slice *m* '(100 900) '(:all)))
(defcase "D06_slice_step_rows"
  "(vt-slice m '(0 1000 3) '(:all))"
  (vt-slice *m* '(0 1000 3) '(:all)))
(defcase "D07_slice_step_cols"
  "(vt-slice m '(:all) '(0 1000 7))"
  (vt-slice *m* '(:all) '(0 1000 7)))
(defcase "D08_slice_both_step"
  "(vt-slice m '(100 200 5) '(0 500 2))"
  (vt-slice *m* '(100 200 5) '(0 500 2)))
(defcase "D09_slice_int_index"
  "(vt-slice m '(500) '(:all))"
  (vt-slice *m* '(500) '(:all)))
(defcase "D10_slice_neg_index"
  "(vt-slice m '(-1) '(:all))"
  (vt-slice *m* '(-1) '(:all)))
(defcase "D11_slice_newa"
  "(vt-slice m '(:newa) '(:all) '(:all))"
  (vt-slice *m* '(:newa) '(:all) '(:all)))
(defcase "D12_slice_elli"
  "(vt-slice m3 '(:elli) '(0 50))"
  (vt-slice *m3* '(:elli) '(0 50)))
(defcase "D13_slice_neg_range"
  "(vt-slice m '(-200 -100) '(:all))"
  (vt-slice *m* '(-200 -100) '(:all)))
(defcase "D14_slice_reverse_cols"
  "(vt-slice m '(:all) '(nil nil -1))"
  (vt-slice *m* '(:all) '(nil nil -1)))
(defcase "D15_slice_reverse_step3"
  "(vt-slice m '(:all) '(nil nil -3))"
  (vt-slice *m* '(:all) '(nil nil -3)))

;;; ---------------------------------------------------------------------------
;;; 6. E 组：split / vsplit / hsplit / dsplit（多输出）
;;; ---------------------------------------------------------------------------

(defcase "E01_split4_ax0"
  "(vt-split m 4 :axis 0)"
  (vt-split *m* 4 :axis 0))
(defcase "E02_split3_ax0"
  "(vt-split m 3 :axis 0)"
  (vt-split *m* 3 :axis 0))
(defcase "E03_split_indices_ax0"
  "(vt-split m '(250 750) :axis 0)"
  (vt-split *m* '(250 750) :axis 0))
(defcase "E04_split5_ax1"
  "(vt-split m 5 :axis 1)"
  (vt-split *m* 5 :axis 1))
(defcase "E05_split_m3_ax2"
  "(vt-split m3 4 :axis 2)"
  (vt-split *m3* 4 :axis 2))
(defcase "E06_vsplit4"
  "(vt-vsplit m 4)"
  (vt-vsplit *m* 4))
(defcase "E07_vsplit_indices"
  "(vt-vsplit m '(300 700))"
  (vt-vsplit *m* '(300 700)))
(defcase "E08_hsplit2"
  "(vt-hsplit m 2)"
  (vt-hsplit *m* 2))
(defcase "E09_hsplit_1d"
  "(vt-hsplit (vt-arange 1000000) 4)"
  (vt-hsplit (vt-arange 1000000) 4))
(defcase "E10_dsplit2"
  "(vt-dsplit m3 2)"
  (vt-dsplit *m3* 2))
(defcase "E11_dsplit4"
  "(vt-dsplit m3 4)"
  (vt-dsplit *m3* 4))

;;; ---------------------------------------------------------------------------
;;; 7. F 组：stack / vstack / hstack / dstack / concatenate / concat（拼接）
;;; ---------------------------------------------------------------------------

(defcase "F01_stack0"
  "(vt-stack 0 (vt-narrow m 0 0 4) (vt-narrow m 0 4 8))"
  (vt-stack 0 (vt-narrow *m* 0 0 4) (vt-narrow *m* 0 4 8)))
(defcase "F02_stack1"
  "(vt-stack 1 (vt-narrow m 0 0 4) (vt-narrow m 0 4 8))"
  (vt-stack 1 (vt-narrow *m* 0 0 4) (vt-narrow *m* 0 4 8)))
(defcase "F03_stack_neg1"
  "(vt-stack -1 (vt-narrow m 0 0 4) (vt-narrow m 0 4 8))"
  (vt-stack -1 (vt-narrow *m* 0 0 4) (vt-narrow *m* 0 4 8)))
(defcase "F04_stack_three"
  "(vt-stack 0 (vt-narrow m 0 0 4) (vt-narrow m 0 4 8) (vt-narrow m 0 8 12))"
  (vt-stack 0 (vt-narrow *m* 0 0 4) (vt-narrow *m* 0 4 8) (vt-narrow *m* 0 8 12)))
(defcase "F05_vstack"
  "(vt-vstack (vt-narrow m 0 0 4) (vt-narrow m 0 4 8))"
  (vt-vstack (vt-narrow *m* 0 0 4) (vt-narrow *m* 0 4 8)))
(defcase "F06_vstack_1d"
  "(vt-vstack (vt-arange 1000) (vt-arange 1000 :start 1000))"
  (vt-vstack (vt-arange 1000) (vt-arange 1000 :start 1000)))
(defcase "F07_hstack"
  "(vt-hstack (vt-narrow m 1 0 4) (vt-narrow m 1 4 8))"
  (vt-hstack (vt-narrow *m* 1 0 4) (vt-narrow *m* 1 4 8)))
(defcase "F08_hstack_1d"
  "(vt-hstack (vt-arange 1000) (vt-arange 1000 :start 1000))"
  (vt-hstack (vt-arange 1000) (vt-arange 1000 :start 1000)))
(defcase "F09_dstack"
  "(vt-dstack (vt-narrow m 0 0 4) (vt-narrow m 0 4 8))"
  (vt-dstack (vt-narrow *m* 0 0 4) (vt-narrow *m* 0 4 8)))
(defcase "F10_concatenate_ax0"
  "(vt-concatenate 0 (vt-narrow m 0 0 4) (vt-narrow m 0 4 8))"
  (vt-concatenate 0 (vt-narrow *m* 0 0 4) (vt-narrow *m* 0 4 8)))
(defcase "F11_concatenate_ax1"
  "(vt-concatenate 1 (vt-narrow m 1 0 4) (vt-narrow m 1 4 8))"
  (vt-concatenate 1 (vt-narrow *m* 1 0 4) (vt-narrow *m* 1 4 8)))
(defcase "F12_concat_neg_axis"
  "(vt-concat -1 (vt-narrow m 1 0 4) (vt-narrow m 1 4 8))"
  (vt-concat -1 (vt-narrow *m* 1 0 4) (vt-narrow *m* 1 4 8)))
(defcase "F13_concatenate_noncontig"
  "(vt-concatenate 0 (vt-transpose m) (vt-transpose m))"
  (vt-concatenate 0 (vt-transpose *m*) (vt-transpose *m*)))
(defcase "F14_concatenate_three"
  "(vt-concatenate 0 (vt-narrow m 0 0 4) (vt-narrow m 0 4 8) (vt-narrow m 0 8 12))"
  (vt-concatenate 0 (vt-narrow *m* 0 0 4) (vt-narrow *m* 0 4 8) (vt-narrow *m* 0 8 12)))
(defcase "F15_concatenate_interleaved"
  "(vt-concatenate 1 (vt-slice m '(0 1000 2) '(:all)) (vt-slice m '(1 1000 2) '(:all)))"
  (vt-concatenate 1 (vt-slice *m* '(0 1000 2) '(:all))
                    (vt-slice *m* '(1 1000 2) '(:all))))

;;; ---------------------------------------------------------------------------
;;; 8. G 组：repeat / tile / pad / broadcast-to / contiguous
;;; ---------------------------------------------------------------------------

(defcase "G01_repeat_ax0"
  "(vt-repeat (vt-narrow m 0 0 4) 2 :axis 0)"
  (vt-repeat (vt-narrow *m* 0 0 4) 2 :axis 0))
(defcase "G02_repeat_ax1"
  "(vt-repeat (vt-narrow m 0 0 4) 3 :axis 1)"
  (vt-repeat (vt-narrow *m* 0 0 4) 3 :axis 1))
(defcase "G03_repeat_list"
  "(vt-repeat (vt-narrow m 0 0 4) '(0 1 2 3) :axis 0)"
  (vt-repeat (vt-narrow *m* 0 0 4) '(0 1 2 3) :axis 0))
(defcase "G04_repeat_flat"
  "(vt-repeat (vt-narrow m 0 0 4) 2)"
  (vt-repeat (vt-narrow *m* 0 0 4) 2))
(defcase "G05_tile_scalar"
  "(vt-tile (vt-narrow m 0 0 4) 2)"
  (vt-tile (vt-narrow *m* 0 0 4) 2))
(defcase "G06_tile_2x3"
  "(vt-tile (vt-narrow m 0 0 4) '(2 3))"
  (vt-tile (vt-narrow *m* 0 0 4) '(2 3)))
(defcase "G07_tile_3d_reps"
  "(vt-tile (vt-narrow m 0 0 4) '(2 1 2))"
  (vt-tile (vt-narrow *m* 0 0 4) '(2 1 2)))
(defcase "G08_pad_constant1"
  "(vt-pad (vt-narrow m 0 0 4) 1)"
  (vt-pad (vt-narrow *m* 0 0 4) 1))
(defcase "G09_pad_perdim"
  "(vt-pad (vt-narrow m 0 0 4) '((2 1) (3 0)))"
  (vt-pad (vt-narrow *m* 0 0 4) '((2 1) (3 0))))
(defcase "G10_pad_edge"
  "(vt-pad (vt-narrow m 0 0 4) 3 :mode :edge)"
  (vt-pad (vt-narrow *m* 0 0 4) 3 :mode :edge))
(defcase "G11_pad_wrap"
  "(vt-pad (vt-narrow m 0 0 4) 3 :mode :wrap)"
  (vt-pad (vt-narrow *m* 0 0 4) 3 :mode :wrap))
(defcase "G12_pad_reflect"
  "(vt-pad (vt-narrow m 0 0 4) 3 :mode :reflect)"
  (vt-pad (vt-narrow *m* 0 0 4) 3 :mode :reflect))
(defcase "G13_pad_symmetric"
  "(vt-pad (vt-narrow m 0 0 4) 3 :mode :symmetric)"
  (vt-pad (vt-narrow *m* 0 0 4) 3 :mode :symmetric))
(defcase "G14_pad_constant_value"
  "(vt-pad (vt-narrow m 0 0 4) 2 :constant-values -1)"
  (vt-pad (vt-narrow *m* 0 0 4) 2 :constant-values -1))
(defcase "G15_broadcast_col"
  "(vt-broadcast-to (vt-reshape (vt-narrow m 1 0 1) '(1000 1)) '(1000 1000))"
  (vt-broadcast-to (vt-reshape (vt-narrow *m* 1 0 1) '(1000 1)) '(1000 1000)))
(defcase "G16_broadcast_row"
  "(vt-broadcast-to (vt-narrow m 0 0 1) '(1000 1000))"
  (vt-broadcast-to (vt-narrow *m* 0 0 1) '(1000 1000)))
(defcase "G17_broadcast_3d"
  "(vt-broadcast-to (vt-reshape (vt-narrow m 1 0 1) '(1 1000 1)) '(2 1000 1000))"
  (vt-broadcast-to (vt-reshape (vt-narrow *m* 1 0 1) '(1 1000 1)) '(2 1000 1000)))
(defcase "G18_broadcast_scalar"
  "(vt-broadcast-to (vt-from-sequence '(7)) '(1000 1000))"
  (vt-broadcast-to (vt-from-sequence '(7)) '(1000 1000)))
(defcase "G19_contiguous_transposed"
  "(vt-contiguous (vt-transpose m))"
  (vt-contiguous (vt-transpose *m*)))
(defcase "G20_contiguous_strided"
  "(vt-contiguous (vt-slice m '(0 1000 2) '(:all)))"
  (vt-contiguous (vt-slice *m* '(0 1000 2) '(:all))))
(defcase "G21_contiguous_already"
  "(vt-contiguous m)"
  (vt-contiguous *m*))
;; 位移取模为 0（shift=0 / shift 为轴长整数倍）：值不变，且结果不与输入
;; 共享底层内存（别名语义由 test/contract-regression-test.lisp 断言）。
(defcase "G22_roll_zero_axis0"
  "(vt-roll m 0 :axis 0)"
  (vt-roll *m* 0 :axis 0))
(defcase "G23_roll_axislen_multiple"
  "(vt-roll m 1000 :axis 0)  ;; mod 1000 = 0"
  (vt-roll *m* 1000 :axis 0))

;;; ---------------------------------------------------------------------------
;;; 9. H 组：triu / tril / diagonal / flatten-to-nested
;;; ---------------------------------------------------------------------------

(defcase "H01_triu"
  "(vt-triu m)"
  (vt-triu *m*))
(defcase "H02_triu_k1"
  "(vt-triu m :k 1)"
  (vt-triu *m* :k 1))
(defcase "H03_tril"
  "(vt-tril m)"
  (vt-tril *m*))
(defcase "H04_tril_k-1"
  "(vt-tril m :k -1)"
  (vt-tril *m* :k -1))
(defcase "H05_diagonal"
  "(vt-diagonal m)"
  (vt-diagonal *m*))
(defcase "H06_diagonal_off_pos"
  "(vt-diagonal m :offset 5)"
  (vt-diagonal *m* :offset 5))
(defcase "H07_diagonal_off_neg"
  "(vt-diagonal m :offset -5)"
  (vt-diagonal *m* :offset -5))
(defcase "H08_diagonal_batch"
  "(vt-diagonal m3)"
  (vt-diagonal *m3*))
(defcase "H09_flatten_to_nested_2d"
  "(vt-from-sequence (vt-flatten-to-nested '(1000 1000) (vt-data m)))"
  (vt-from-sequence (vt-flatten-to-nested '(1000 1000) (vt-data *m*))))
(defcase "H10_flatten_to_nested_3d"
  "(vt-from-sequence (vt-flatten-to-nested '(100 100 100) (vt-data m3)))"
  (vt-from-sequence (vt-flatten-to-nested '(100 100 100) (vt-data *m3*))))
;; batch 对角线显式指定轴（numpy 语义；默认仍取末两轴）
(defcase "H11_diagonal_axis1_0_axis2_1"
  "(vt-diagonal m3 :axis1 0 :axis2 1)  ;; = np.diagonal(m3, axis1=0, axis2=1)"
  (vt-diagonal *m3* :axis1 0 :axis2 1))
(defcase "H12_diagonal_axis1_0_axis2_2"
  "(vt-diagonal m3 :axis1 0 :axis2 2)  ;; = np.diagonal(m3, axis1=0, axis2=2)"
  (vt-diagonal *m3* :axis1 0 :axis2 2))
(defcase "H13_diagonal_axis1_1_axis2_2_off3"
  "(vt-diagonal m3 :offset 3 :axis1 1 :axis2 2)  ;; = np.diagonal(m3, 3, 1, 2)"
  (vt-diagonal *m3* :offset 3 :axis1 1 :axis2 2))

;;; ---------------------------------------------------------------------------
;;; 10. I 组：append / insert / delete
;;; ---------------------------------------------------------------------------

(defcase "I01_append_ax0"
  "(vt-append m (vt-narrow m 0 0 3) :axis 0)"
  (vt-append *m* (vt-narrow *m* 0 0 3) :axis 0))
(defcase "I02_append_ax1"
  "(vt-append m (vt-narrow m 1 0 3) :axis 1)"
  (vt-append *m* (vt-narrow *m* 1 0 3) :axis 1))
(defcase "I03_append_flat"
  "(vt-append m (vt-from-sequence '(1 2 3)) :axis nil)"
  (vt-append *m* (vt-from-sequence '(1 2 3)) :axis nil))
(defcase "I04_append_flat_views"
  "(vt-append (vt-narrow m 0 0 2) (vt-narrow m 0 2 5) :axis nil)"
  (vt-append (vt-narrow *m* 0 0 2) (vt-narrow *m* 0 2 5) :axis nil))
(defcase "I05_insert_ax0"
  "(vt-insert m 5 (vt-narrow m 0 0 3) :axis 0)"
  (vt-insert *m* 5 (vt-narrow *m* 0 0 3) :axis 0))
;; I06：axis=1 插入单列，values 形状 (1, n)。moveaxis(0→1) 对 (1,n) 退化为
;;      平铺，clvt 与 numpy 在此形状下一致。
(defcase "I06_insert_ax1"
  "(vt-insert m 7 (vt-narrow m 0 0 1) :axis 1)"
  (vt-insert *m* 7 (vt-narrow *m* 0 0 1) :axis 1))
;; I15：axis=1 且 values 含多行（numnew>1）。
;;   此前用行主序 reshape 顶替 moveaxis(values,0,axis)：当元素总数恰好相同
;;   （此处 values=(2,1000) 与块 (1000,2) 均 2000 个）时形状校验通过，从而
;;   静默产生与 numpy 不同的结果；现施加 moveaxis，插入列为 i（第 7 列）
;;   与 1000+i（第 8 列）。
(defcase "I15_insert_ax1_moveaxis"
  "(vt-insert m 7 (vt-narrow m 0 0 2) :axis 1)"
  (vt-insert *m* 7 (vt-narrow *m* 0 0 2) :axis 1))
(defcase "I07_insert_flat"
  "(vt-insert (vt-narrow m 0 0 10) 3 (vt-from-sequence '(100 200)) :axis nil)"
  (vt-insert (vt-narrow *m* 0 0 10) 3 (vt-from-sequence '(100 200)) :axis nil))
(defcase "I08_insert_multi_pos"
  "(vt-insert (vt-narrow m 0 0 6) '(1 3) (vt-narrow m 0 0 2) :axis 0)"
  (vt-insert (vt-narrow *m* 0 0 6) '(1 3) (vt-narrow *m* 0 0 2) :axis 0))
(defcase "I09_delete_ax0"
  "(vt-delete m 5 :axis 0)"
  (vt-delete *m* 5 :axis 0))
(defcase "I10_delete_ax1"
  "(vt-delete m 5 :axis 1)"
  (vt-delete *m* 5 :axis 1))
(defcase "I11_delete_list"
  "(vt-delete m '(1 3 5) :axis 0)"
  (vt-delete *m* '(1 3 5) :axis 0))
(defcase "I12_delete_neg"
  "(vt-delete m -1 :axis 0)"
  (vt-delete *m* -1 :axis 0))
(defcase "I13_delete_flat"
  "(vt-delete m 0 :axis nil)"
  (vt-delete *m* 0 :axis nil))
(defcase "I14_delete_flat_list"
  "(vt-delete (vt-narrow m 0 0 2) '(0 1 2) :axis nil)"
  (vt-delete (vt-narrow *m* 0 0 2) '(0 1 2) :axis nil))

;;; ---------------------------------------------------------------------------
;;; 11. J 组：take / put / where / argwhere / nonzero / choose / select /
;;;           extract / searchsorted / digitize / bincount / astype
;;; ---------------------------------------------------------------------------

(defcase "J01_take_ax0"
  "(vt-take m (vt-from-sequence '(0 5 999 500) :dtype :int64) :axis 0)"
  (vt-take *m* (vt-from-sequence '(0 5 999 500) :dtype :int64) :axis 0))
(defcase "J02_take_ax1"
  "(vt-take m (vt-from-sequence '(0 999) :dtype :int64) :axis 1)"
  (vt-take *m* (vt-from-sequence '(0 999) :dtype :int64) :axis 1))
(defcase "J03_take_flat"
  "(vt-take m (vt-from-sequence '(0 999999 500000 123) :dtype :int64))"
  (vt-take *m* (vt-from-sequence '(0 999999 500000 123) :dtype :int64)))
(defcase "J04_take_duplicates"
  "(vt-take m (vt-from-sequence '(3 3 3) :dtype :int64) :axis 0)"
  (vt-take *m* (vt-from-sequence '(3 3 3) :dtype :int64) :axis 0))
(defcase "J05_put_flat"
  "(let ((c (vt-copy (vt-narrow m 0 0 3)))) (vt-put c (vt-from-sequence '(0 2999 1500) :dtype :int64) (vt-from-sequence '(-1 -2 -3))) c)"
  (let ((c (vt-copy (vt-narrow *m* 0 0 3))))
    (vt-put c (vt-from-sequence '(0 2999 1500) :dtype :int64)
              (vt-from-sequence '(-1 -2 -3)))
    c))
(defcase "J06_put_duplicate_indices"
  "(let ((c (vt-copy (vt-narrow m 0 0 3)))) (vt-put c (vt-from-sequence '(0 0 1) :dtype :int64) (vt-from-sequence '(10 20 30))) c)"
  (let ((c (vt-copy (vt-narrow *m* 0 0 3))))
    (vt-put c (vt-from-sequence '(0 0 1) :dtype :int64)
              (vt-from-sequence '(10 20 30)))
    c))
(defcase "J07_where_cond_only_gt"
  "(vt-where (vt-> m 999000))"
  (vt-where (vt-> *m* 999000)))
(defcase "J08_where_cond_only_eq"
  "(vt-where (vt-= m 123456))"
  (vt-where (vt-= *m* 123456)))
(defcase "J09_where_three_arg"
  "(vt-where (vt-> m 500000) m (vt-flip m :axis 1))"
  (vt-where (vt-> *m* 500000) *m* (vt-flip *m* :axis 1)))
(defcase "J10_where_scalar_branch"
  "(vt-where (vt-< m 5) m -1)"
  (vt-where (vt-< *m* 5) *m* -1))
(defcase "J11_argwhere_eq"
  "(vt-argwhere (vt-= m 123456))"
  (vt-argwhere (vt-= *m* 123456)))
(defcase "J12_argwhere_gt"
  "(vt-argwhere (vt-> m 999000))"
  (vt-argwhere (vt-> *m* 999000)))
(defcase "J13_argwhere_range"
  "(vt-argwhere (vt-logical-and (vt->= m 250000) (vt-< m 250010)))"
  (vt-argwhere (vt-logical-and (vt->= *m* 250000) (vt-< *m* 250010))))
(defcase "J14_nonzero_eq"
  "(vt-nonzero (vt-= m 123456))"
  (vt-nonzero (vt-= *m* 123456)))
(defcase "J15_nonzero_gt"
  "(vt-nonzero (vt-> m 999000))"
  (vt-nonzero (vt-> *m* 999000)))
(defcase "J16_nonzero_3d"
  "(vt-nonzero (vt-= m3 500000))"
  (vt-nonzero (vt-= *m3* 500000)))
(defcase "J17_choose"
  "(vt-choose (list m (vt-flip m :axis 1)) (vt-astype (vt-> m 500000) :int64))"
  (vt-choose (list *m* (vt-flip *m* :axis 1))
             (vt-astype (vt-> *m* 500000) :int64)))
(defcase "J18_choose_clip"
  "(vt-choose (list m (vt-flip m :axis 1)) (vt-- (vt-astype (vt-> m 500000) :int64) 1) :mode :clip)"
  (vt-choose (list *m* (vt-flip *m* :axis 1))
             (vt-- (vt-astype (vt-> *m* 500000) :int64) 1)
             :mode :clip))
(defcase "J19_select"
  "(vt-select (list (vt-> m 999000) (vt-< m 5)) (list m (vt-flip m :axis 1)) :default -1)"
  (vt-select (list (vt-> *m* 999000) (vt-< *m* 5))
             (list *m* (vt-flip *m* :axis 1))
             :default -1))
(defcase "J20_extract_eq"
  "(vt-extract (vt-= m 123456) m)"
  (vt-extract (vt-= *m* 123456) *m*))
(defcase "J21_extract_gt"
  "(vt-extract (vt-> m 999000) m)"
  (vt-extract (vt-> *m* 999000) *m*))
(defcase "J22_searchsorted_left"
  "(vt-searchsorted base (vt-from-sequence '(0 500000 999999 123456)) :side :left)"
  (vt-searchsorted *base* (vt-from-sequence '(0 500000 999999 123456)) :side :left))
(defcase "J23_searchsorted_right"
  "(vt-searchsorted base (vt-from-sequence '(0 500000 999999 123456)) :side :right)"
  (vt-searchsorted *base* (vt-from-sequence '(0 500000 999999 123456)) :side :right))
(defcase "J24_searchsorted_vector"
  "(vt-searchsorted (vt-arange 1000) (vt-narrow m 0 0 5))"
  (vt-searchsorted (vt-arange 1000) (vt-narrow *m* 0 0 5)))
(defcase "J25_digitize"
  "(vt-digitize (vt-from-sequence '(-1 0 500 999.5 1000 1000000 2000000)) (vt-arange 1001 :start 0 :step 1000))"
  (vt-digitize (vt-from-sequence '(-1 0 500 999.5 1000 1000000 2000000))
               (vt-arange 1001 :start 0 :step 1000)))
(defcase "J26_digitize_right"
  "(vt-digitize (vt-from-sequence '(-1 0 500 999.5 1000 1000000 2000000)) (vt-arange 1001 :start 0 :step 1000) :right t)"
  (vt-digitize (vt-from-sequence '(-1 0 500 999.5 1000 1000000 2000000))
               (vt-arange 1001 :start 0 :step 1000) :right t))
(defcase "J27_bincount"
  "(vt-bincount (vt-astype (vt-rem (vt-* (vt-arange 100000 :dtype :int64) 37) 20) :int64))"
  (vt-bincount (vt-astype (vt-rem (vt-* (vt-arange 100000 :dtype :int64) 37) 20)
                          :int64)))
(defcase "J28_bincount_minlength"
  "(vt-bincount (vt-from-sequence '(1 3 5) :dtype :int64) :minlength 10)"
  (vt-bincount (vt-from-sequence '(1 3 5) :dtype :int64) :minlength 10))
(defcase "J29_astype_int32"
  "(vt-astype m :int32)"
  (vt-astype *m* :int32))
(defcase "J30_astype_float32"
  "(vt-astype m :float32)"
  (vt-astype *m* :float32))
(defcase "J31_astype_int16"
  "(vt-astype (vt-narrow m 0 0 2) :int16)"
  (vt-astype (vt-narrow *m* 0 0 2) :int16))
(defcase "J32_astype_noncontig"
  "(vt-astype (vt-transpose m) :float32)"
  (vt-astype (vt-transpose *m*) :float32))
(defcase "J33_astype_int64"
  "(vt-astype m :int64)"
  (vt-astype *m* :int64))

;;; ---------------------------------------------------------------------------
;;; 12. K 组：多算子组合（多组合测试）
;;; ---------------------------------------------------------------------------

(defcase "K01_combo_trans_reshape_contig"
  "(vt-contiguous (vt-reshape (vt-transpose m) '(500 2000)))"
  (vt-contiguous (vt-reshape (vt-transpose *m*) '(500 2000))))
(defcase "K02_combo_slice_trans_flatten"
  "(vt-flatten (vt-transpose (vt-slice m '(0 1000 2) '(:all))))"
  (vt-flatten (vt-transpose (vt-slice *m* '(0 1000 2) '(:all)))))
(defcase "K03_combo_reshape_narrow_trans"
  "(vt-transpose (vt-narrow (vt-reshape m '(100 100 100)) 1 10 90))"
  (vt-transpose (vt-narrow (vt-reshape *m* '(100 100 100)) 1 10 90)))
(defcase "K04_combo_pad_flip_roll"
  "(vt-roll (vt-flip (vt-pad (vt-narrow m 0 0 4) 2) :axis 1) 3 :axis 1)"
  (vt-roll (vt-flip (vt-pad (vt-narrow *m* 0 0 4) 2) :axis 1) 3 :axis 1))
(defcase "K05_combo_stack_reshape"
  "(vt-reshape (vt-stack 0 (vt-narrow m 0 0 2) (vt-narrow m 0 2 4)) '(4 1000))"
  (vt-reshape (vt-stack 0 (vt-narrow *m* 0 0 2) (vt-narrow *m* 0 2 4)) '(4 1000)))
(defcase "K06_combo_split_vstack_roundtrip"
  "(apply #'vt-vstack (vt-split (vt-narrow m 0 0 8) 4 :axis 0))"
  (apply #'vt-vstack (vt-split (vt-narrow *m* 0 0 8) 4 :axis 0)))
(defcase "K07_combo_trans_swapaxes"
  "(vt-swapaxes (vt-transpose m3 '(2 0 1)) 0 1)"
  (vt-swapaxes (vt-transpose *m3* '(2 0 1)) 0 1))
(defcase "K08_combo_tile_transpose"
  "(vt-transpose (vt-tile (vt-narrow m 0 0 4) '(1 2)))"
  (vt-transpose (vt-tile (vt-narrow *m* 0 0 4) '(1 2))))
(defcase "K09_combo_broadcast_concat"
  "(vt-concatenate 1 (vt-broadcast-to (vt-reshape (vt-narrow m 1 0 1) '(1000 1)) '(1000 500)) (vt-narrow m 1 0 500))"
  (vt-concatenate 1 (vt-broadcast-to (vt-reshape (vt-narrow *m* 1 0 1) '(1000 1))
                                     '(1000 500))
                    (vt-narrow *m* 1 0 500)))
(defcase "K10_combo_pad_flatten"
  "(vt-flatten (vt-pad (vt-narrow m 0 0 4) 1))"
  (vt-flatten (vt-pad (vt-narrow *m* 0 0 4) 1)))
(defcase "K11_combo_rot90_reshape_flat"
  "(vt-reshape (vt-rot90 m :k 1) '(1000000))"
  (vt-reshape (vt-rot90 *m* :k 1) '(1000000)))
(defcase "K12_combo_concat_repeat_narrow"
  "(vt-repeat (vt-concatenate 0 (vt-narrow m 0 0 2) (vt-narrow m 0 998 1000)) 2 :axis 0)"
  (vt-repeat (vt-concatenate 0 (vt-narrow *m* 0 0 2) (vt-narrow *m* 0 998 1000))
             2 :axis 0))
(defcase "K13_combo_where_flatten"
  "(vt-flatten (vt-where (vt-> m 999000) m 0))"
  (vt-flatten (vt-where (vt-> *m* 999000) *m* 0)))
(defcase "K14_combo_squeeze_unsqueeze_roundtrip"
  "(vt-squeeze (vt-unsqueeze m 1) :axis 1)"
  (vt-squeeze (vt-unsqueeze *m* 1) :axis 1))
(defcase "K15_combo_diagonal_take"
  "(vt-take (vt-diagonal m) (vt-from-sequence '(0 500 999) :dtype :int64))"
  (vt-take (vt-diagonal *m*) (vt-from-sequence '(0 500 999) :dtype :int64)))
(defcase "K16_combo_flip_both_axes"
  "(vt-flip (vt-flip m :axis 0) :axis 1)"
  (vt-flip (vt-flip *m* :axis 0) :axis 1))
(defcase "K17_combo_strided_interleave_concat"
  "(vt-concatenate 0 (vt-slice m '(0 1000 2) '(:all)) (vt-slice m '(1 1000 2) '(:all)))"
  (vt-concatenate 0 (vt-slice *m* '(0 1000 2) '(:all))
                    (vt-slice *m* '(1 1000 2) '(:all))))
(defcase "K18_combo_pad_trans_contig"
  "(vt-contiguous (vt-transpose (vt-pad (vt-narrow m 1 0 4) '((0 0) (2 3)))))"
  (vt-contiguous (vt-transpose (vt-pad (vt-narrow *m* 1 0 4)
                                       '((0 0) (2 3))))))
(defcase "K19_combo_4d_perm_reshape"
  "(vt-reshape (vt-transpose (vt-reshape m '(10 100 1000)) '(1 2 0)) '(1000 1000))"
  (vt-reshape (vt-transpose (vt-reshape *m* '(10 100 1000)) '(1 2 0)) '(1000 1000)))
(defcase "K20_combo_insert_delete_roundtrip"
  "(vt-delete (vt-insert (vt-narrow m 0 0 6) 2 (vt-narrow m 0 0 1) :axis 0) 2 :axis 0)"
  (vt-delete (vt-insert (vt-narrow *m* 0 0 6) 2 (vt-narrow *m* 0 0 1) :axis 0)
             2 :axis 0))

;;; ---------------------------------------------------------------------------
;;; 13. 运行器：落盘 + 往返自检 + manifest
;;; ---------------------------------------------------------------------------

(defun save-case-outputs (name tensors)
  "把 TENSORS 落盘，返回 (npy-info lisp-info)。"
  (let ((npy-info nil) (lisp-info nil) (total 0))
    (loop for i from 0 for tt in tensors do
      (incf total (vt-size tt))
      (let* ((suffix (if (= (length tensors) 1) "" (format nil ".~D" i)))
             (npy (out-path (format nil "~A~A.npy" name suffix)))
             (sexp (out-path (format nil "~A~A.sexp" name suffix))))
        (vt-save tt npy :as :npy)
        ;; npy 往返校验
        (let ((back (vt-load npy :as :npy)))
          (unless (vt-array-equal back tt)
            (error "npy 往返不一致: ~A~A" name suffix)))
        (push (list :file (file-namestring npy)
                    :shape (vt-shape tt)
                    :dtype (dtype-name tt)) npy-info)
        ;; 小结果额外写 :lisp 原生格式并往返校验
        (when (<= (vt-size tt) *lisp-max-elems*)
          (vt-save tt sexp :as :lisp)
          (let ((back (vt-load sexp :as :lisp)))
            (unless (vt-array-equal back tt)
              (error ":lisp 往返不一致: ~A~A" name suffix)))
          (push (list :file (file-namestring sexp)
                      :shape (vt-shape tt)) lisp-info))))
    (list (nreverse npy-info) (nreverse lisp-info) total)))

(defun write-manifest (records)
  (let ((path (out-path "manifest.json")))
    (with-open-file (o path :direction :output :if-exists :supersede)
      (format o "{~%  \"generator\": \"vt-manip-interop-test.lisp\",~%")
      (format o "  \"sbcl\": ~A,~%" (json-str (lisp-implementation-version)))
      (format o "  \"clvt_version\": ~A,~%"
              (json-str (or (asdf:component-version (asdf:find-system :clvt)) "unknown")))
      (format o "  \"base_expression\": ~A,~%"
              (json-str "(vt-reshape (vt-arange 1000000) '(1000 1000))"))
      (format o "  \"base_shape\": [1000, 1000],~%")
      (format o "  \"base_dtype\": \"float64\",~%")
      (format o "  \"case_count\": ~D,~%" (length records))
      (format o "  \"cases\": [~%")
      (loop for rec in records for first = t then nil do
        (destructuring-bind (name form npy lisp size) rec
          (unless first (format o ",~%"))
          (format o "    {\"name\": ~A,~%" (json-str name))
          (format o "     \"form\": ~A,~%" (json-str form))
          (format o "     \"elems\": ~D,~%" size)
          (format o "     \"outputs\": [~{~A~^, ~}],~%"
                  (mapcar (lambda (e)
                            (format nil "{\"file\": ~A, \"shape\": ~A, \"dtype\": ~A}"
                                    (json-str (getf e :file))
                                    (json-int-list (getf e :shape))
                                    (json-str (getf e :dtype))))
                          npy))
          (format o "     \"lisp_outputs\": [~{~A~^, ~}]}"
                  (mapcar (lambda (e)
                            (format nil "{\"file\": ~A, \"shape\": ~A}"
                                    (json-str (getf e :file))
                                    (json-int-list (getf e :shape))))
                          lisp))))
      (format o "~%  ]~%}~%"))
    path))

(defun run-cases ()
  (ensure-directories-exist (out-path "manifest.json"))
  (let ((records nil) (n-ok 0) (failures nil))
    (dolist (c (reverse *cases*))
      (destructuring-bind (name form thunk) c
        (handler-case
            (let* ((tensors (normalize-results (funcall thunk)))
                   (saved (save-case-outputs name tensors)))
              (push (list name form (first saved) (second saved) (third saved)) records)
              (incf n-ok)
              (format t "~&[saved] ~A  (~D 输出, ~D 元素, ~A)~%"
                      name (length tensors) (third saved) form))
          (error (e)
            (push (list name form (format nil "~A" e)) failures)
            (format t "~&[FAIL ] ~A  *** ~A ***  ~A~%" name e form)))))
    (let ((mp (write-manifest (nreverse records))))
      (format t "~&~%生成成功用例: ~D / ~D~%manifest: ~A~%"
              n-ok (length *cases*) mp))
    (when failures
      (format t "~&~%===== 生成阶段失败的用例 (~D) =====~%" (length failures))
      (dolist (f failures)
        (format t "~&  - ~A : ~A~%     形式: ~A~%" (first f) (third f) (second f))))
    n-ok))

;;; ---------------------------------------------------------------------------
;;; 14. Lisp 端独立自检：手算 oracle + 错误契约（不依赖 NumPy）
;;; ---------------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)

(defmacro chk (label expected form)
  "断言 FORM 与 EXPECTED 相等（用 equal）。"
  `(let ((got ,form) (exp ,expected))
     (cond ((equal got exp)
            (incf *pass*)
            (format t "~&  PASS ~A~%" ,label))
           (t (incf *fail*)
              (format t "~&  FAIL ~A  期望 ~S 实得 ~S~%" ,label exp got)))))

(defmacro chk-error (label &body body)
  "断言 BODY 触发 error。"
  `(handler-case
       (progn ,@body
              (incf *fail*)
              (format t "~&  FAIL ~A  (应报错但未报错)~%" ,label))
     (error () (incf *pass*) (format t "~&  PASS ~A (正确报错)~%" ,label))))

(defun self-check ()
  (format t "~&~%===== Lisp 端独立自检（oracle / 错误契约）=====~%")

  ;; -- 元素位置 oracle：m[i][j] = i*1000 + j --
  (chk "oracle: m[7][5] == 7005" 7005.0d0 (vt-ref *m* 7 5))
  (chk "oracle: m[999][999] == 999999" 999999.0d0 (vt-ref *m* 999 999))
  (chk "oracle: transpose[7][5] == m[5][7] == 5007"
       5007.0d0 (vt-ref (vt-transpose *m*) 7 5))
  (chk "oracle: flip(ax0)[0][5] == m[999][5] == 999005"
       999005.0d0 (vt-ref (vt-flip *m* :axis 0) 0 5))
  (chk "oracle: flip(ax1)[5][0] == m[5][999] == 5999"
       5999.0d0 (vt-ref (vt-flip *m* :axis 1) 5 0))
  (chk "oracle: roll(ax0,5)[0][0] == m[995][0] == 995000"
       995000.0d0 (vt-ref (vt-roll *m* 5 :axis 0) 0 0))
  (chk "oracle: roll(ax1,-3)[0][0] == m[0][3] == 3"
       3.0d0 (vt-ref (vt-roll *m* -3 :axis 1) 0 0))
  (chk "oracle: narrow(0,100,900)[0][0] == m[100][0] == 100000"
       100000.0d0 (vt-ref (vt-narrow *m* 0 100 900) 0 0))
  (chk "oracle: swapaxes[3][4] == m[4][3] == 4003"
       4003.0d0 (vt-ref (vt-swapaxes *m* 0 1) 3 4))
  (chk "oracle: reverse cols [0][0] == m[0][999] == 999"
       999.0d0 (vt-ref (vt-slice *m* '(:all) '(nil nil -1)) 0 0))
  ;; rot90: result[i][j] = m[j][N-1-i]
  (chk "oracle: rot90[0][0] == m[0][999] == 999"
       999.0d0 (vt-ref (vt-rot90 *m*) 0 0))
  (chk "oracle: rot90[1][2] == m[2][998] == 2998"
       2998.0d0 (vt-ref (vt-rot90 *m*) 1 2))
  ;; diagonal: diag[i] = m[i][i]
  (chk "oracle: diagonal[5] == 5005" 5005.0d0 (vt-ref (vt-diagonal *m*) 5))
  (chk "oracle: diagonal(off=5)[0] == m[0][5] == 5"
       5.0d0 (vt-ref (vt-diagonal *m* :offset 5) 0))
  (chk "oracle: diagonal(off=-5)[0] == m[5][0] == 5000"
       5000.0d0 (vt-ref (vt-diagonal *m* :offset -5) 0))
  ;; vt-diagonal 的 batch 约定：对「最后两轴」取对角线（同 torch.diagonal）；
  ;; numpy 的 np.diagonal(a) 默认 axis1=0/axis2=1 属历史特例，batch 下与 clvt 不同。
  (chk "oracle: diagonal(m3)[0][1] == m3[0][1][1] == 101.0 (最后两轴约定)"
       101.0d0 (vt-ref (vt-diagonal *m3*) 0 1))
  ;; triu/tril
  (chk "oracle: triu[0][999] == m[0][999] == 999"
       999.0d0 (vt-ref (vt-triu *m*) 0 999))
  (chk "oracle: triu[999][0] == 0" 0.0d0 (vt-ref (vt-triu *m*) 999 0))
  (chk "oracle: tril[999][999] == 999999" 999999.0d0 (vt-ref (vt-tril *m*) 999 999))
  (chk "oracle: tril[0][999] == 0" 0.0d0 (vt-ref (vt-tril *m*) 0 999))
  ;; take / where
  (chk "oracle: take flat '500000' == 500000"
       500000.0d0 (vt-ref (vt-take *m* (vt-from-sequence '(500000) :dtype :int64)) 0))
  ;; 单参 where = np.where(cond) = np.asarray(cond).nonzero()
  ;; 返回按轴分组元组（每轴一个索引张量），不是 argwhere 的 (n, rank)
  (chk "oracle: single-arg where = 按轴分组元组 (= vt-nonzero)"
       '((123) (456))
       (mapcar #'vt-to-list (vt-where (vt-= *m* 123456))))
  (chk "oracle: argwhere 仍为 (n,rank)"
       '((123 456)) (vt-to-list (vt-argwhere (vt-= *m* 123456))))
  (chk "oracle: argwhere count(gt 999000) == 999"
       999 (first (vt-shape (vt-argwhere (vt-> *m* 999000)))))
  ;; item / ref
  (chk "oracle: item(slice m 500 700) == 500700"
       500700.0d0 (vt-item (vt-slice *m* '(500) '(700))))
  ;; 连续性 oracle
  (chk "oracle: contiguous-p m == T" t (vt-contiguous-p *m*))
  (chk "oracle: contiguous-p transpose == NIL" nil (vt-contiguous-p (vt-transpose *m*)))
  (chk "oracle: contiguous-p contiguous(transpose) == T"
       t (vt-contiguous-p (vt-contiguous (vt-transpose *m*))))
  (chk "oracle: contiguous-p ravel == T" t (vt-contiguous-p (vt-ravel *m*)))
  (chk "oracle: contiguous-p slice step == NIL"
       nil (vt-contiguous-p (vt-slice *m* '(0 1000 2) '(:all))))
  ;; 形状 oracle
  (chk "oracle: squeeze(1,1000,1000,1) -> (1000 1000)"
       '(1000 1000) (vt-shape (vt-squeeze (vt-reshape *m* '(1 1000 1000 1)))))
  (chk "oracle: unsqueeze m 1 -> (1000 1 1000)"
       '(1000 1 1000) (vt-shape (vt-unsqueeze *m* 1)))
  (chk "oracle: split m 3 -> shapes (334 333 333)"
       '(334 333 333)
       (mapcar (lambda (x) (first (vt-shape x))) (vt-split *m* 3 :axis 0)))
  (chk "oracle: vsplit m 4 -> 4 x (250 1000)"
       '((250 1000) (250 1000) (250 1000) (250 1000))
       (mapcar #'vt-shape (vt-vsplit *m* 4)))
  (chk "oracle: dstack shape (4 1000 2)"
       '(4 1000 2)
       (vt-shape (vt-dstack (vt-narrow *m* 0 0 4) (vt-narrow *m* 0 4 8))))
  (chk "oracle: tile '(2 3)' shape (8 3000)"
       '(8 3000) (vt-shape (vt-tile (vt-narrow *m* 0 0 4) '(2 3))))
  (chk "oracle: pad 1 shape (6 1002)"
       '(6 1002) (vt-shape (vt-pad (vt-narrow *m* 0 0 4) 1)))
  (chk "oracle: broadcast col: b[i][j] == m[i][0]"
       t (let ((b (vt-broadcast-to (vt-reshape (vt-narrow *m* 1 0 1) '(1000 1))
                                   '(1000 1000))))
           (and (= (vt-ref b 0 0) 0.0d0)
                (= (vt-ref b 5 0) (vt-ref b 5 999) 5000.0d0)
                (= (vt-ref b 999 0) (vt-ref b 999 999) 999000.0d0))))
  ;; flatten-to-nested 往返
  (chk "oracle: flatten-to-nested 2d 往返"
       t (vt-array-equal *m* (vt-from-sequence (vt-flatten-to-nested
                                                '(1000 1000) (vt-data *m*)))))
  ;; setf vt-slice 写回
  (chk "oracle: setf vt-slice 写回"
       '(777.0d0 777.0d0) (let ((c (vt-copy (vt-narrow *m* 0 0 6))))
                            (setf (vt-slice c '(1 3) '(0 2)) (vt-full '(2 2) 777))
                            (list (vt-ref c 1 0) (vt-ref c 2 1))))
  ;; narrow 越界按 numpy 切片语义 clip（v0.4.2 起），不报错
  (chk "oracle: narrow end 越界被 clip 到 dim"
       '(1000 1000) (vt-shape (vt-narrow *m* 0 0 1001)))

  ;; -- 错误契约 --
  (chk-error "err: vt-view 非连续"
             (vt-view (vt-transpose *m*) '(1000000)))
  (chk-error "err: vt-item 多元素"
             (vt-item (vt-narrow *m* 0 0 3)))
  (chk-error "err: vt-reshape 元素数不符"
             (vt-reshape *m* '(999 1000)))
  (chk-error "err: vt-narrow start>end"
             (vt-narrow *m* 0 900 100))
  (chk-error "err: vt-slice step=0"
             (vt-slice *m* '(0 100 0) '(:all)))
  (chk-error "err: vt-slice 索引越界"
             (vt-slice *m* '(1000) '(:all)))
  (chk-error "err: vt-vsplit 不整除"
             (vt-vsplit *m* 3))
  (chk-error "err: vt-stack 形状不一致"
             (vt-stack 0 (vt-narrow *m* 0 0 3) (vt-narrow *m* 0 0 4)))
  (chk-error "err: vt-concatenate 秩不同"
             (vt-concatenate 0 *m* (vt-ravel *m*)))
  (chk-error "err: vt-concatenate 非轴维度不一致"
             (vt-concatenate 0 (vt-narrow *m* 0 0 3) (vt-narrow *m* 1 0 4)))
  (chk-error "err: vt-tile 负 reps"
             (vt-tile (vt-narrow *m* 0 0 3) '(2 -1)))
  (chk-error "err: vt-broadcast-to 不可广播"
             (vt-broadcast-to (vt-narrow *m* 0 0 3) '(1000 1000)))
  (chk-error "err: vt-choose raise 越界"
             (vt-choose (list *m* (vt-flip *m* :axis 1))
                        (vt-full '(1000 1000) 5 :dtype :int64)))
  (chk-error "err: vt-repeat 列表长度不符"
             (vt-repeat (vt-narrow *m* 0 0 3) '(1 2) :axis 1))

  (format t "~&~%Lisp 自检：PASS=~D FAIL=~D~%" *pass* *fail*)
  (values *pass* *fail*))

;;; ---------------------------------------------------------------------------
;;; 15. 入口
;;; ---------------------------------------------------------------------------

(defun main ()
  (format t "~%=== clvt 形状/拼接/切片 互操作测试生成器 ===~%")
  (format t "SBCL ~A / clvt ~A~%" (lisp-implementation-version)
          (or (asdf:component-version (asdf:find-system :clvt)) "?"))
  (format t "基准: (vt-reshape (vt-arange 1000000) '(1000 1000))  shape=~A dtype=~A~%"
          (vt-shape *m*) (vt-dtype *m*))
  (format t "输出目录: ~A~%" *out-dir*)
  (let ((n (run-cases)))
    (multiple-value-bind (p f) (self-check)
      (declare (ignore p))
      (if (zerop f)
          (progn (format t "~%=== 生成完成：~D 个用例，自检全部通过 ===~%" n)
                 (sb-ext:exit :code 0))
          (progn (format t "~%=== 存在 ~D 个自检失败 ===~%" f)
                 (sb-ext:exit :code 1))))))

(main)
