import PIL.Image

import mydb
import numpy as np
import numpy
import tools
if __name__ == "__main__":
    # 测试连接 detect_types=sqlite3.PARSE_DECLTYPES !!
    conn = mydb.getConn()
    cur = conn.cursor()
    x = np.random.rand(1,512).astype(numpy.float32)
    pth = "face4lsh.jpg"
    img = PIL.Image.open(pth)
    frame = numpy.asarray(img)
    print(frame.dtype)
    emb, _ =tools.FaceVerifier.get_emb_and_cropped_from_np(frame)
    print(emb.dtype)
    print(emb.shape)
    with conn:
    #插入
        conn.execute("insert into user_authentication (username, password, faceVector)"
                    "values ('4', '4', :arr)" , {"arr":emb})