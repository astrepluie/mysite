# flask 프레임워크 로드
from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
from invest import Quant
from database import MyDB

# MyDB class 생성
mydb = MyDB(
    _host = 'kmminna.mysql.pythonanywhere-services.com',
    _port = 3306,
    _user = 'kmminna',
    _pw = 'rain1234',
    _db_name = 'kmminna$default'
)

# Flask 클래스 생성
# 생성자 함수에 필요한 인자 : 파일의 이름
app = Flask(__name__)

# 네비게이터 -> 특정한 주소로 요청이 들어왔을 때 함수와 연결
# route() 함수 인자값 : root url 뒤에 붙는 주소
@app.route('/')
def index() :
    return render_template('login.html')

@app.route('/main', methods = ['post'])
def main():
    # 유저가 보낸 데이터를 변수에 저장
    # get 방식으로 보낸 데이터 : request.args
    # 포스트 방식으로 보낸 데이터 : request.form
    user_id = request.form['input_id']
    user_pass = request.form['input_pass']
    print(f"id : {user_id}, pass : {user_pass}")
    # 유저가 입력한 아이디와 비밀번호를 DB server에 해당 데이터가 존재하는가?
    login_query = """
        select * from `user`
        where (`id` = %s) and (`password` = %s) 
    """
    result_sql = mydb.sql_query(login_query, user_id, user_pass)
    # result_sql이 존재한다면 -> 로그인이 성공
    if result_sql :
        return render_template('index.html')
    # 존재하지 않으면 -> 로그인이 실패
    else :
        # 로그인 페이지를 보여주는 주소로 이동
        return redirect('/')

@app.route('/signup')
def signup() :
    return render_template('id_check.html')

# id값 중복체크하는 주소를 생성
@app.route('/id_check')
def id_check() :
    # 유저가 보낸 아이디를 변수에 저장
    user_id = request.args['input_id']
    id_check_query = "select * from `user` where `id` = %s"
    result_sql = mydb.sql_query(id_check_query, user_id)
    print(result_sql)
    # result_sql이 존재한다면 -> 회원가입 불가
    if result_sql :
        return redirect('/signup')
    else :
        # 사용 가능한 아이디
        return render_template('signup2.html', id = user_id)

@app.route('/user_insert', methods = ['post'])
def user_insert() :
    # 유저가 보낸 데이터 3개
    user_id = request.form['input_id']
    user_pass = request.form['input_pass']
    user_name = request.form['input_name']
    # DB server에 데이터를 insert
    insert_query = "insert into `user` values (%s, %s, %s)"
    mydb.sql_query(insert_query, user_id, user_pass, user_name)
    mydb.commit_db()
    return render_template('signup3.html')
    

@app.route('/invest')
def invest() :
    input_code = request.args['code']
    input_start_time = f"{request.args['s_year']}-{request.args['s_month']}-{request.args['s_day']}"
    input_end_time = f"{request.args['e_year']}-{request.args['e_month']}-{request.args['e_day']}"
    input_kind = request.args['kind']
    print(input_code, input_start_time, input_end_time, input_kind)
    # input_code를 이용해서 csv 파일 로드
    df = pd.read_csv(f'home/kmminna/mysite/csv/{input_code}.csv')
    df.rename(columns = {'날짜' : 'Date'}, inplace = True)
    quant = Quant(df, _start = input_start_time, _end = input_end_time, _col = 'Close')
    if input_kind == 'bnh' :
        result, rtn = quant.buyandhold()
    elif input_kind == 'boll' :
        result, rtn = quant.bollinger()
    elif input_kind == 'hall' :
        result, rtn = quant.halloween()
    elif input_kind == 'mm' :
        result, rtn = quant.momentum()
    result.reset_index(inplace = True)
    result = result.loc[ result['rtn'] != 1 , : ]
    cols = list(result.columns)
    value = result.to_dict('records')
    x = list(result['Date'])
    y = list(result['acc_rtn'])
    res_data = {
        'columns' : cols,
        'values' : value,
        'axis_x' : x,
        'axis_y' : y 
    }
    return res_data 

