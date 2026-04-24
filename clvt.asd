;;;; clvt.asd

(asdf:defsystem #:clvt
  :description "common lisp vector tensor libaray"
  :author "xizang123321@gmail.com>"
  :license  "MIT"
  :version "0.0.1"
  :serial t
  :components ((:file "package")
               (:file "vector-tensor")
	       (:file "vector-tensor-print")
	       (:file "einsum")
	       (:file "statistics-aggregation")
	       (:file "arithmetic-operation")
	       (:file "compatible-functions")
	       (:file "activation")
	       (:file "loss-probability")
	       (:file "linalg-basic")
	       (:file "linalg-solve")
	       (:file "linalg-advanced")))
