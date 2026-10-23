;;;; clvt.asd

(asdf:defsystem #:clvt
  :description "common lisp vector tensor library"
  :author "xizang123321@gmail.com"
  :license  "MIT"
  :version "0.4.13"
  :serial t
  ;; ------------------------------------------------------------------
  ;; 读取格式锁定（**不要删**）
  ;;
  ;; 本库源码里存在大量「裸浮点字面量」（`0.0`、`0.5`、`180.0` …），
  ;; 它们的精度由 **read 时的 `*read-default-float-format*`** 决定，
  ;; 而不是由代码意图决定。CL 标准只要求实现默认值为 SINGLE-FLOAT，
  ;; 但用户可以（而且不少科学计算用户确实会）在自己的 .sbclrc 里
  ;;     (setf *read-default-float-format* 'double-float)
  ;;
  ;; 一旦如此，`(complex 0.0 0.0)`（complex64 的零值）会被读成
  ;; `(complex 0.0d0 0.0d0)`，于是把 double 复数填进 complex64 数组：
  ;;   · SBCL 的 vector-fill 对每个元素类型都装了带类型声明的转换器
  ;;     `(lambda (sb-c::item) (declare (type (complex single-float) item)) ...)`，
  ;;     直接抛
  ;;       The value #C(0.0 0.0) is not of type (COMPLEX SINGLE-FLOAT)
  ;;       when binding SB-C::ITEM
  ;;   · 更糟的是在 safety 0 的路径上它会**静默写入错误位模式**。
  ;;
  ;; 因此这里把读取格式钉死为 SINGLE-FLOAT（库内约定：裸字面量 = 单精度，
  ;; 双精度一律写 `d0` 后缀）。ASDF 的 :AROUND-COMPILE 包裹 compile-file /
  ;; load，read 发生在其动态范围内，故对全部组件生效。
  ;; 注意：**已有的 fasl 缓存不会自动失效** —— 换过该变量的用户需要
  ;; `(asdf:load-system :clvt :force t)` 或清掉 ~/.cache/common-lisp 后重编。
  ;; ------------------------------------------------------------------
  :around-compile (lambda (next)
                    (let ((*read-default-float-format* 'single-float))
                      (funcall next)))
  :depends-on (#+sbcl #:sb-simd)
  :components ((:file "src/package")
               (:file "src/util")
               (:file "src/iterator")
               (:file "src/nan")
               (:file "src/dtype")
               (:file "src/core")
               (:file "src/dispatch")
               ;; 复数**标量内核**（第十三轮）：纯函数，只吃数、吐数。
               ;; 必须早于所有归属文件 —— elementwise / extensions3 /
               ;; reduce-stats 的单一 DEFUN 分派直接调用这些内核。
               (:file "src/complex-kernels")
               (:file "src/parcontract")
               (:file "src/map-reduce")
               (:file "src/io")
               (:file "src/creation")
               (:file "src/manip")
               (:file "src/indexing")
               (:file "src/join")
               (:file "src/elementwise")
	       (:file "src/elementwise1")
	       (:file "src/elementwise2")
               (:file "src/reduce-stats")
               (:file "src/setops")
               (:file "src/random")
               (:file "src/linalg")
               #+sbcl (:file "src/simd-matmul")
               (:file "src/nn")
               (:file "src/rotate")
               (:file "src/extensions")
               (:file "src/extensions2")
               (:file "src/extensions3")
               ;; 复数语义层（CONVENTIONS §16）：必须**最后**加载 ——
               ;; 它按「输入是否含复数」对公开函数做前置分派，
               ;; 加载时捕获各函数的实数原实现。
               (:file "src/complex")
               ;; 复数数据搬运层（§16.18，第二轮）：转换/拼接/剪切/集合/
               ;; 缺失值/缩并。依赖 complex 的 %CX-* 内核，故紧随其后。
               (:file "src/complex-ops")
               ;; 复数线性代数层（§16.20，第三轮）：LU/det/solve/inv/
               ;; Cholesky/QR/SVD/厄米特特征值/pinv/lstsq/matrix-rank。
               ;; 依赖前面两层的 %CX-* 工具与 vt-matmul，故排在最后。
               (:file "src/complex-linalg")))
