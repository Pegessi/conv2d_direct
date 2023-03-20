# conv2d_direct

直接计算conv2d的cuda算子实现

详细介绍参考：https://zhuanlan.zhihu.com/p/613538649

## 编译

需要确保本机有cuda和cudnn

`nvcc main.cu -o test -lcudnn`

如果找不到cudnn请手动指定目录"# conv2d_direct" 
