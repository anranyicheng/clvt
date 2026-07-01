;;;; clvt.asd

(asdf:defsystem #:clvt
  :description "common lisp vector tensor libaray"
  :author "xizang123321@gmail.com>"
  :license  "MIT"
  :version "0.0.1"
  :serial t
  :components ((:file "package")
	       (:file "utili")
	       (:file "nan")
               (:file "vector-tensor")
	       (:file "map-reduce")
	       (:file "random")
	       (:file "vector-tensor-print")
	       (:file "einsum")
	       (:file "statistics-aggregation")
	       (:file "arithmetic-operation")
	       (:file "compatible-functions")
	       (:file "rotate")
	       (:file "activation")
	       (:file "loss-probability")
	       (:file "linalg")))
