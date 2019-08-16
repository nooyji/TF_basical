"""
텐서플로우 배치 처리

텐서플로우에서 파일에서 데이터를 읽은 후에, 배치 처리로 placeholder에서 읽는 예제를 설명한다.
텐서의 shape의 차원과 세션의 실행 시점 개념 확실히
"""

import tensorflow as tf
import numpy as np
import sys

TRAINING_FILE = '' # path 설정

# 배치 처리 코드
# 데이터를 텐서플로우에서 읽어서 배치로 placeholder에 feeding하는 코드
# 먼저 read_data는 csv 파일에서 데이터를 읽어서 파싱을 한 후 column을 year, flight, time으로 리턴하는 함수이다.

def read_data(file_name):
    try:
        csv_file = tf.train.string_input_producer([file_name], name = 'filename_queue')
        textReader = tf.TextLineReader()
        _,line = textReader.read(csv_file)
        # 라인 단위로 데이터를 읽어서 
        year, flight, time = tf.decode_csv(line, record_defaults = [[1900], [""],[0]], field_delim = ','])
        # 파싱하는 작업
    except:
        print("Unexpected error", sys.exc_info()[0])
        exit()
    return year, flight, time
    # column을 return

"""
string_input_producer를 통해서 파일명들을 큐잉해서 하나씩 읽는데, 여기서는 편의상 하나의 파일만 읽도록 하였는데,
여러개의 파일을 병렬로 처리하고자 한다면, [file_name] 부분에 리스트 형으로 여러개의 파일 목록을 지정해주면 된다.
다음 각 파일을 TextReader를 이용하여 라인 단위로 읽은 후 
decode_csv를 이용하여, ","로 분리된 컬럼을 각각 읽어서 year, flight, time에 저장하여 리턴하였다.
"""

# 다음 함수는 read_data_batch라는 함수인데, 앞에서 정의한 read_data 함수를 호출하여, 
# 읽어들인 year, flight, time을 배치로 묶어서 리턴하는 함수이다.

def read_data_batch(file_name, batch_size=10):
    year, flight, time = read_data(file_name)
    batch_year, batch_flight, batch_time = tf.train.batch([year, flight, time], batch_size = batch_size)

    return batch_year, batch_flight, batch_time

"""
tf.train.batch 함수가 배치로 묶어서 리턴을 하는 함수인데, batch로 묶고자 하는 tensor들을 인자로 준 다음에,
batch_size(한번에 묶어서 리턴하고자 하는 텐서들의 갯수)를 정해주면 된다.

위의 예제에서는 batch_size를 10으로 해줬기 때문에, batch_year = [1900, 1901, ..., 1909]와 같은 형태로
10개의 년도를 하나의 텐서에 묶어서 리턴해준다.
즉, 입력 텐서의 shape가 [x, y, z]일 경우 tf.train.batch를 통한 출력은 [batch_size, x, y, z]가 된다. (핵심)
"""

# 메인 코드

def main():

    print("start session")
    # coornator(?) 위에 코드가 있어야 한다.
    # 데이터를 집어 넣기 전에 미리 그래프가 만들어져 있어야 함
    batch_year, batch_flight, batch_time = read_data_batch(TRAINING_FILE)
    year = tf.placeholder(tf.int32, [None, ])
    flight = tf.placeholder(tf.string, [None, ])
    time = tf.placeholder(tf.int32, [None, ])

    tt = time * 10

"""
tt = time * 10 이라는 공식을 실행하기 위해서 time이라는 값을 읽어서 피딩하는 예제인데
먼저 read_data_batch를 이용하여 데이터를 읽는 그래프를 생성한다.
이 때 주의해야 할 점은 이 함수를 수행한다고 해서 바로 데이터를 읽기 시작하는 것이 아니라,
데이터의 흐름을 정의하는 그래프만 생성된다는 것을 주의하자

다음으로는 year, flight, time placeholder를 정의한다.
year, flight, time은 0차원의 scalar 텐서이지만, 값이 연속적으로 들어오기 때문에 [None, ]으로 정의한다.
즉, year = [1900, 1901, 1902, 1903, ...] 형태이기 때문에 1차원 Vector형태의 shape로 [None, ]으로 정의한다.
Placeholder들에 대한 정의가 끝났으면 세션을 정의하고 데이터를 읽어들이기 위한 Queue runner를 수행한다.
앞의 과정까지 텐서 그래프를 다 그렸고 이 그래프 값을 부어넣기 위해서 Queue runner를 수행한 것이다.
"""
    with tf.Session() as sess:
        try:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            # Queue runner를 실행하였기 때문에 데이터가 데이터 큐로 들어가기 시작하고,
            # 이 큐에 들어간 데이터를 읽어들이기 위해서, 세션을 실행한다.
            y__, f__, t__ = sess.run([batch_year, batch_flight, batch_time])
            print(sess.run(tt, feed_dict = {time:t__}))
            # 세션을 실행하면, batch_year, batch_flight, batch_time 값을 읽어서 y__, f__, t__ 변수에 각각
            # 집어 넣은 다음에 t__ 값을 tt 공식의 time 변수에 feeding하여 값을 계산한다.

            # 모든 작업이 끝났으면 아래와 같이 Queue runner를 정지시킨다.
            coord.request_stop()
            coord.join(threads)
        except:
            print("Unexpected error: ", sys.exc_info()[0])

main()
