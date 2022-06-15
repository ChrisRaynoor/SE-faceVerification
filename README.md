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
