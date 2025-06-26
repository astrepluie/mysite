import pandas as pd
import numpy as np
from datetime import datetime

def create_YM(_df, col = 'Adj Close') :
    # 데이터프레임 복사본 생성
    df = _df.copy()
    # 컬럼에 Date가 존재하면 index로 변경
    if 'Date' in df.columns :
        df.set_index('Date', inplace = True)
    # index를 시계열로 변경
    df.index = pd.to_datetime(df.index)
    # 결측치, 무한대 제거, 기준이 되는 컬럼을 제외하고 나머지 제거
    flag = (df.isin([np.nan, np.inf, -np.inf]).any(axis=1))
    df = df.loc[-flag, [col] ]
    # STD-YM 컬럼을 생성, 인덱스에서 연-월 데이터를 추출해서 대입
    df['STD-YM'] = df.index.strftime('%Y-%m')
    return df

def create_last_month(_df, _start = '2010-01-01', _end = datetime.now(), _momentum = 12) :
    # 기준 컬럼 명 변수에 저장
    col = _df.columns[0]
    # 월말의 기준 : STD-YM이 현재와 다음 행의 데이터가 다른 경우
    flag = (_df['STD-YM'] != _df.shift(-1)['STD-YM'])
    df = _df.loc[flag, : ]
    # 전월의 기준이 되는 컬럼의 데이터를 BF1 컬럼에 대입, 결측치 0으로 채움
    df['BF1'] = df.shift(1)[col].fillna(0)
    # _momentum 개월 전 데이터를 생성해서 BF2 컬럼에 대입, 결측치 0으로 채움
    df['BF2'] = df.shift(_momentum)[col].fillna(0)
    # 투자 시작 시간과 종료 시간으로 필터링
    df = df.loc[_start : _end, : ]
    return df

def create_rtn(_df1, _df2, _start = '2010-01-01', _end = datetime.now(), _score = 1) :
    # _df1의 복사본 생성
    df = _df1.copy()
    df = df.loc[_start : _end, : ]
    # trade 컬럼과 rtn 컬럼 생성
    df['trade'] = ''
    df['rtn'] = 1

    for idx in _df2.index :
        signal = ''
        # 절대 모멘텀 인덱스 계산
        momentum_index = (_df2.loc[idx, 'BF1'] / _df2.loc[idx, 'BF2']) - _score
        # 조건식 : 모멘텀 인덱스가 0보다 크고 무한대가 아닌 경우 구매내역 추가
        if (momentum_index > 0) & (momentum_index != np.inf) :
            signal = 'buy'
        df.loc[idx, 'trade'] = signal
    # 기준 컬럼명 가져오기
    col = df.columns[0]
    # 수익률 계산
    for idx in df.index :
        if (df.shift().loc[idx, 'trade'] == '') & (df.loc[idx, 'trade'] == 'buy') :
            buy = df.loc[idx, col]
            print(f"매수일 : {idx}, 매수가 : {buy}")
        elif (df.shift().loc[idx, 'trade'] == 'buy') & (df.loc[idx, 'trade'] == '') :
            sell = df.loc[idx, col]
            rtn = sell/buy
            df.loc[idx, 'rtn'] = rtn
            print(f"매도일 : {idx}, 매도가 : {sell}, 수익률 : {rtn}") 
    # 누적수익률 계산
    df['acc_rtn'] = df['rtn'].cumprod()
    acc_rtn = df.iloc[-1,-1]
    return df, acc_rtn