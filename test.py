# 测试数据库连接和存取numpy浮点
import sqlite3
from settings import *
import numpy as np
import io

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
x = np.random.rand(256)
#0.9668762  0.33231055
print(x)
print(x.dtype)
# 测试连接 detect_types=sqlite3.PARSE_DECLTYPES !!
conn = sqlite3.connect(DB_ADDRESS,detect_types=sqlite3.PARSE_DECLTYPES)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
# 插入
# cur.execute("insert into user_authentication (username, password, faceVector)"
#             "values ('testDirectIns', 'password', :arr)" , {"arr":x})
conn.commit()
cur = conn.execute("select user_id from user_authentication where username='ntestDirectIns'")
#row = cur.fetchone()
print({cur.fetchone() is None})