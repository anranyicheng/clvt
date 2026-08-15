;;;; clvt.asd

(asdf:defsystem #:clvt
  :description "common lisp vector tensor library"
  :author "xizang123321@gmail.com"
  :license  "MIT"
  :version "0.2.0"
  :serial t
  :components ((:file "src/package")
               (:file "src/dtype")
               (:file "src/util")
               (:file "src/nan")
               (:file "src/core")
               (:file "src/iterator")
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
               (:file "src/nn")
               (:file "src/rotate")
               (:file "src/extensions")))
