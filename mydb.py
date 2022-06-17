# ./mydb.py
# 进行对数据库的基本操作

from settings import *
import sqlite3
import numpy as np
import io
import logging
from logging import debug
# 向sqlite3注册ndarray
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)
# end 注册

# 连接数据库返回conn
def getConn():
    try:
        conn = sqlite3.connect(DB_ADDRESS,detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        raise e
def closeConn(conn):
    try:
        conn.close()
    except sqlite3.DatabaseError as e:
        pass
# 关闭数据库连接
# def closeConn(conn):
#     try:
#         conn.close()
#     except Exception as e:
#         raise e
#
# def execQuery(sql):
#     try:
#         conn = sqlite3.connect(DB_ADDRESS,detect_types=sqlite3.PARSE_DECLTYPES)
#         conn.row_factory = sqlite3.Row
#         cur = conn.cursor()
#         cur.execute(sql)
#         cur.close()
#         conn.commit()
#         conn.close()
#
# debug test
if __name__ == "__main__":
    # 测试连接 detect_types=sqlite3.PARSE_DECLTYPES !!
    conn = sqlite3.connect(DB_ADDRESS,detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    x = np.random.rand(256)
    # 使用例: with conn
    # Using the nonstandard execute(), executemany() and executescript() methods of the Connection object,
    # your code can be written more concisely because you don’t have to create
    # the (often superfluous) Cursor objects explicitly. Instead, the Cursor objects are created implicitly
    # and these shortcut methods return the cursor objects. This way, you can execute a SELECT statement
    # and iterate over it directly using only a single call on the Connection object.
    # and auto commit (the "with" block is a transaction)
    with conn:
    #插入
        conn.execute("insert into user_authentication (username, password, faceVector)"
                    "values ('testDirectIns', 'password', :arr)" , {"arr":x})
        name = "testIndirectIns"
        password = "password2"
        conn.execute("insert into user_authentication (username, password, faceVector)"
                "values (:uname, :pw, :arr)" , {"arr":x, 'pw':password, 'uname':name})


    # conn.commit()
    # # 测试读取
    # # cur.execute("select faceVector from user_authentication")
    # # row = cur.fetchone()
    # # print(type(row[0]))
    # # print(row[0])
    # # print(row[0].dtype)
    #
    # cur.execute("select * from user_authentication")
    # row = cur.fetchone()
    # print(type(row))
    # arr = row['faceVector']
    # print(type(arr))
    # print(arr)

# todo 数据库操作优化，可忽略
# 采用每次操作都连接和断开数据库的模式
# 会不会重复创建连接
# 连接断开的处理
# 需考虑连接失败的情况

# or简单一点，只进行一次connect
# 不考虑连接异常
# 所有窗口退出时调用一个善后函数(Qt有预定义吗？),包括close的执行
# 一些操作还需commit -- commit失败的判断和处理

# 如果使用占位符形式，如何