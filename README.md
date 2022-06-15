项目文件结构说明:
./resources 存放qt使用的资源文件
xxx.ui QtDesigner设计文件,需要pyuic转换
ui_xxx.py pyuic转换生成,不要在此实现交互逻辑
run_xxx.py 对应窗口的逻辑实现
database.sqlite 数据库文件
models.py userModel实现与数据库的直接交互
mydb.py 数据库工具
settings.py 定义全局设置,如常量等
tools.py 预计用来实现可调用的人脸验证等模块

记得不要上传IDE项目文件到仓库

---
_环境配置_
https://blog.csdn.net/ifeng12358/article/details/103002102
python 版本:
3.6

qt配置:
pip install：
pyqt5 （注:5.15.x）
pyqt5-tools（注:5.15.x）

pycharm配置 external tools:
qt-desinger
pyui
pyrcc

配置sqlite（自带）

其他:pytroch, cv2 等

----
在dev分支上开发，仅push到dev分支，main分支上仅存储可发布版：
### 首次拉取分支
> 当从远程库clone时，默认情况下，你只能看到本地的master分支。可以用git branch命令看看：
> `$ git branch `
> `* master `
> 现在，要在dev分支上开发，就必须创建远程origin的dev分支到本地，于是用这个命令创建本地dev分支：
> `$ git checkout -b dev origin/dev `
> 现在，就可以在dev上继续修改，然后，时不时地把dev分支push到远程：

--小结：
使用`$ git switch -c dev origin/dev` 等
