# Noise


## 更新日志
**2023.5.25更新：**
* 新建项目

## 文件说明
#### <pre>main.py</pre>
训练程序
> 可选参数：
```
--input: 原始路径，默认为根目录下的input文件夹
--output: 输出路径，默认为根目录下的outpit文件夹
--noise: 噪声类型，可选['salt_and_pepper', 'gaussian', 'blur']，默认为salt_and_pepper
```
#### <pre>noise.py</pre>
噪声的实现
#### <pre>config.yaml</pre>
噪声的相关参数