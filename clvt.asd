;;;; clvt.asd

(asdf:defsystem #:clvt
  :description "common lisp vector tensor library"
  :author "xizang123321@gmail.com"
  :license  "MIT"
  :version "0.2.0"
  :serial t
  :depends-on (#+sbcl #:sb-simd)
  :components ((:file "src/package")
	       (:file "src/util")
	       (:file "src/iterator")
	       (:file "src/nan")
               (:file "src/dtype")
               (:file "src/core")              
               (:file "src/map-reduce")
               (:file "src/io")
               (:file "src/creation")
               (:file "src/manip")
               (:file "src/indexing")
               (:file "src/join")
               (:file "src/elementwise")
               (:file "src/reduce-stats")
               (:file "src/setops")
               (:file "src/random")
               (:file "src/linalg")
	       #+sbcl (:file "src/simd-matmul")
               (:file "src/nn")
               (:file "src/rotate")
               (:file "src/extensions")
	       (:file "src/extensions2")))
