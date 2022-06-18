# ./models.py
# 实现与数据库直接交互的模型
import tempfile
import numpy
import tools
import numpy as np
import mydb
import sqlite3
from PyQt5.QtCore import QObject, pyqtSignal
from logging import debug

class User(QObject):
    # 定义该类会发出的信号
    # faceVector_get_signal = pyqtSignal(list)
    # 默认初始化一个空用户
    def __init__(self, uid = None, username = None):
        super(User, self).__init__()
        self.uid = uid
        self.username = username
        self.loggedIn = False
        self.tmpDir = None # 由tempfile类生成的用于该用户的临时文件路径，应在logout时删除
        # tip:
        # 一个user对象可能没有对应的数据库记录,是否存在要调用其id查一次

    # 尝试登录并保存信息到一个实例，此时实例变量不一定有效
    def login(self, username, password):
        # 数据库连接错误另外处理
        conn = mydb.getConn()
        # 下面的try主要保证close被执行，因为没查到关于自动释放的文档
        try:
            with conn:
                cur = conn.execute("select user_id from user_authentication where username=:a_name and password=:pw",
                                   {"a_name": username, "pw":password})
                row = cur.fetchone()
                if row is None:
                    return False
                # 登录成功
                self.uid = row['user_id']
                self.username = username
                self.loggedIn = True
                self.tmpDir = tempfile.TemporaryDirectory()
                return True
        # 可用于处理with conn内的错误
        # except sqlite3.DatabaseError as e:
        finally:
            mydb.closeConn(conn)
    # 登出
    def logout(self):
        self.uid = None
        self.username = None
        self.loggedIn = False
        self.tmpDir = None
    # 判断用户存在
    @staticmethod
    def isUsernameExist(username):
        # 数据库连接错误另外处理
        conn = mydb.getConn()
        # 下面的try主要保证close被执行，因为没查到关于自动释放的文档
        try:
            with conn:
                cur = conn.execute("select user_id from user_authentication where username=:a_name",
                                   {"a_name": username})
                row = cur.fetchone()
                if row is None:
                    return False
                return True
        # 可用于处理with conn内的错误
        # except sqlite3.DatabaseError as e:
        finally:
            mydb.closeConn(conn)
    # 验证密码
    def isPasswordCorrect(self, password):
        # 数据库连接错误另外处理
        conn = mydb.getConn()
        # 下面的try主要保证close被执行，因为没查到关于自动释放的文档
        try:
            with conn:
                cur = conn.execute("select password from user_authentication where user_id=:a_uid",
                             {"a_uid":self.uid})
                row = cur.fetchone()
                if row is None:
                    return False
                debug(type(row['password']))
                debug(type(password))
                if row["password"] != password:
                    return False
                return True
        # 可用于处理with conn内的错误
        # except sqlite3.DatabaseError as e:
        finally:
            mydb.closeConn(conn)
    # 录入人脸
    def setFaceVector(self, faceVector):
        '''
        更改数据库中的faceVector
        :param faceVector:
        :return:
        '''
        # 数据库连接错误另外处理
        conn = mydb.getConn()
        # 下面的try主要保证close被执行，因为没查到关于自动释放的文档
        try:
            with conn:
                # 确认用户存在
                if not self.loggedIn:
                    return False
                # cur = conn.execute("select user_id from user_authentication where user_id=:a_uid",
                #              {"a_uid":self.uid})
                # row = cur.fetchone()
                # if row is None:
                #     return False
                # 更新人脸
                cur = conn.execute("update user_authentication set faceVector=:a_vector where user_id=:a_uid",
                                   {"a_vector":faceVector, "a_uid":self.uid})
                return True
        # 可用于处理with conn内的错误
        # except sqlite3.DatabaseError as e:
        finally:
            mydb.closeConn(conn)

    # 获取faceVector
    def getFaceVector(self):
        # 数据库连接错误另外处理
        conn = mydb.getConn()
        # 下面的try主要保证close被执行，因为没查到关于自动释放的文档
        try:
            with conn:
                # 确认用户存在和获取vector
                cur = conn.execute("select faceVector from user_authentication where user_id=:a_uid",
                             {"a_uid":self.uid})
                row = cur.fetchone()
                if row is None:
                    # self.faceVector_get_signal.emit(list())
                    return None
                # self.faceVector_get_signal.emit(row["faceVector"])
                return row["faceVector"]
        # 可用于处理with conn内的错误
        # except sqlite3.DatabaseError as e:
        finally:
            mydb.closeConn(conn)

    # 人脸验证
    def isSameFace(self, Img):
        tools.getEmb()


# debug test
if __name__ == "__main__":
    user = User.createByName("ntestDirectIns")
    conn = mydb.getConn()
    x = np.random.rand(256)
    conn.execute("update user_authentication set faceVector=:n where user_id=0",
                 {"n":x})
    conn.commit()