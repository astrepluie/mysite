import pandas as pd
from datetime import datetime
import numpy as np

def bnh(_df, _start = '2010-01-01', _end = datetime.now(), _col = 'Adj Close') :
    # 복사본 생성
    df = _df.copy()
    # Date가 컬럼에 있다면 Date를 인덱스로 변경
    if 'Date' in df.columns :
        df.set_index('Date', inplace = True)
    # 인덱스 데이터를 시계열로 변경
    df.index = pd.to_datetime(df.index)
    # 결측치, 양의 무한대, 음의 무한대 제거
    flag = df.isin([np.nan, np.inf, -np.inf]).any(axis = 1)
    df = df.loc[-flag, : ]
    # _start, _end 기준 인덱스 필터링, _col 기준 컬럼 필터링
    try : 
        price_df = df.loc[_start : _end , [_col] ]
    except Exception as e :
        print(e)
        print('인자값이 잘못 되었습니다.')
        return 
    # 일별수익률 컬럼 생성하여 pct_change() + 1 데이터 대입
    price_df['rtn'] = (price_df[_col].pct_change() + 1).fillna(1)
    # 누적수익률 계산하여 새로운 컬럼(acc_rtn)에 대입
    price_df['acc_rtn'] = price_df['rtn'].cumprod()
    return price_df, price_df['acc_rtn'][-1]