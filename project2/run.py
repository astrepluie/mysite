import pandas as pd
import warnings
from result import model_result

warnings.filterwarnings('ignore')

# 데이터 선택
data_select = input("데이터를 선택하세요(상장/비상장/ALL) : ")
data_select = data_select.upper()

if data_select in ['상장', '비상장', 'ALL'] :
    df = pd.read_csv(f'data/{data_select}.csv')
else :
    raise ValueError ("데이터를 잘못 선택했습니다.")

feature_cols = ['이익잉여금', '자본', '영업활동으로 인한 현금흐름', '재무활동으로 인한 현금흐름', '투자활동으로 인한 현금흐름', '법인세비용차감전손익']

X_train = df.loc[df['회계년도'] <= '2017/12', feature_cols]
X_test = df.loc[df['회계년도'] > '2017/12', feature_cols]
y_train = df.loc[df['회계년도'] <= '2017/12', 'label']
y_test = df.loc[df['회계년도'] > '2017/12', 'label']

# smote 사용 여부 선택
use_smote = input("smote를 진행할까요?(Y/N)")
use_smote = use_smote.upper()

if use_smote == 'Y' :
    print("smote를 진행합니다.")
    use_SMOTE = True
elif use_smote == 'N' :
    print("smote를 진행하지 않습니다.")
    use_SMOTE = False
else :
    print("선택이 잘못되었습니다. 기본적으로 smote 없이 진행합니다.")
    use_SMOTE = False

# 모델 선택
model_select = input(
    """
        모델을 선택하세요.
        1 : 로지스틱회귀모형 
        2 : RandomForest
        3 : CatBoost
        4 : XGBoost
        5 : LSTM 
        6 : TabNet
    """
)

if model_select == '1' :
    print("모델 선택 : 로지스틱회귀모형")
    from LR import LR_run
    result = LR_run(X_train, y_train, X_test, use_SMOTE)
elif model_select == '2' :
    print("모델 선택 : RandomForest")
    from RF import RF_run
    result = RF_run(X_train, y_train, X_test, use_SMOTE)
elif model_select == '3' :
    print("모델 선택 : CatBoost")
    from CatBoost import Cat_run
    result = Cat_run(X_train, y_train, X_test, use_SMOTE)
elif model_select == '4' :
    print("모델 선택 : XGBoost")
    from XGBoost import xg_run
    result = xg_run(X_train, y_train, X_test, use_SMOTE)
elif model_select == '5' :
    print("모델 선택 : LSTM")
    from LSTM import lstm_run
    result = lstm_run(X_train, y_train, X_test, use_SMOTE)
elif model_select == '6' :
    print("모델 선택 : TabNet")
    from TabNet import tabnet_run
    result = tabnet_run(X_train, y_train, X_test, use_SMOTE)
else :
    raise ValueError ("숫자를 잘못 선택했습니다.")

model_result(result, y_test)