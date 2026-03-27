# clvt
commmon lisp vector tensor library.
这是一个 lisp 张量库，是使用智谱清言为主，DeepSeek 为辅助编写的张量库，
目标就是为 common lisp 生态构建一个简洁而强大的张量计算库。虽然lisp社区拥有
magicl 和 numcl 这两个比较流行的库，但是 magicl 缺乏对高维张量的支持，
numcl 繁重的类型推理，很难理解，numcl 接口和CL语言标准重合，难以论说。
这库的核心是 einsum 计算引擎，以及 vt-map,vt-reduce,其余操作大多数
都是基于这三个核心函数组合完成，易于理解。
## License
MIT

