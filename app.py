import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import altair as alt
import lightgbm

@st.cache_resource
def load_model():
    try:
        with open('models/final_LGBM_MODEL_and_threshold.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['threshold']
    except FileNotFoundError:
        return None, None
    
def preprocess_data(df_raw):
    df = df_raw.copy()
    df['TA_YM'] = pd.to_datetime(df['TA_YM'])
    df.sort_values(by=['ENCODED_MCT', 'TA_YM'], inplace=True)
    if 'MCT_ME_D' in df.columns:
        df = df.drop(columns=['MCT_ME_D'])
    df_cleaned = df.dropna()
    columns_to_drop = [
        'is_crisis',
        'is_closed',
        'ARE_D',
        'MCT_ME_D',
        'TA_YM',
        'LAT',
        'LON',
        'ENCODED_MCT',
        'MCT_BSE_AR',
        'MCT_NM',
        '행정동',
        'MCT_BRD_NUM',
    ]
    X_cols = [col for col in df_cleaned.columns if col not in columns_to_drop]
    X = df_cleaned[X_cols]
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    return X, df_cleaned

def map_feature_name(feature_name, MAPPING_DCT):
    korean_name = MAPPING_DCT.get(feature_name)
    if korean_name:
        return korean_name    
    parts = feature_name.rsplit('_', 1)  
    if len(parts) > 1:
        original_col_name = parts[0]
        dummy_value = parts[1]       
        if original_col_name in CATEGORICAL_COLS:
            original_kor_name = MAPPING_DCT.get(original_col_name)
            return f'{original_kor_name}_{dummy_value}'
    return feature_name

def generate_shap_report(model, X_input, model_features):
    explainer = shap.TreeExplainer(model, model_output='raw')
    shap_values = explainer.shap_values(X_input)  
    shap_values_crisis = shap_values.flatten() 
    proba = model.predict_proba(X_input)[:, 1][0]
    prediction = 1 if proba >= THRESHOLD else 0   
    shap_df = pd.DataFrame({
        'Feature': model_features,
        'SHAP_Value': shap_values_crisis
    }).sort_values(by='SHAP_Value', key=lambda x: np.abs(x), ascending=False)
    top_5_contributing_features = shap_df.head(5).copy()
    top_5_contributing_features['Feature_KOR'] = top_5_contributing_features['Feature'].apply(
        lambda x: map_feature_name(x, MAPPING_DCT)
    )
    chart_df = top_5_contributing_features[['Feature_KOR', 'SHAP_Value']]
    return prediction, proba, top_5_contributing_features, chart_df

def plot_feature_importances(model, mapping_dct):
    importances = pd.Series(model.feature_importances_, index=model.feature_name_)
    
    df_importance = importances[importances > 0].sort_values(ascending=False).head(10).reset_index()
    df_importance.columns = ['Feature', 'Importance']

    df_importance['Feature_KOR'] = df_importance['Feature'].apply(
        lambda x: map_feature_name(x, mapping_dct)
    )

    chart = (
        alt.Chart(df_importance)
        .mark_bar()
        .encode(
            y=alt.Y('Feature_KOR', 
                    title='주요 변수',
                    sort=alt.EncodingSortField(field="Importance", order="descending") 
            ),
            x=alt.X('Importance', 
                    title='중요도',
            ),
            color=alt.value(PASTEL_BLUE),
            tooltip=[
                alt.Tooltip('Feature_KOR', title='변수'),
                alt.Tooltip('Importance', title='중요도', format='.1f')
            ]
        ).interactive()
    )
    return chart





MAPPING_DF = pd.read_csv('data/mapping.csv')
MAPPING_DCT = dict(zip(MAPPING_DF['ENG'], MAPPING_DF['KOR']))
CATEGORICAL_COLS = [
    'HPSN_MCT_ZCD_NM',
    'HPSN_MCT_BZN_CD_NM',
]
PASTEL_RED = '#FF9999'
PASTEL_BLUE = '#99CCFF'
KEY_VARIABLES = [
    'MCT_OPE_MS_CN',
    'RC_M1_SAA',
    'RC_M1_TO_UE_CT',
    'RC_M1_UE_CUS_CN',
    'RC_M1_AV_NP_AT',
]
LGBM_MODEL, THRESHOLD = load_model()





if LGBM_MODEL is None:
    st.stop()
    
if 'current_month_index' not in st.session_state:
    st.session_state.current_month_index = 0

st.set_page_config(
    page_title='경영 위기 조기 경보 시스템',
    page_icon='🚨',
    layout='centered',
)
st.title('🚨 경영 위기 조기 경보 시스템')
st.write('우리 동네 가맹점, 위기 신호를 미리 잡아라!')





st.divider()





uploaded_file = st.file_uploader(
    '📤 분석할 가맹점의 데이터를 올려 주세요',
    type=['csv'],
)





st.divider()





if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    X_processed, df_cleaned = preprocess_data(df_raw)
        
    available_dates = df_cleaned['TA_YM'].sort_values(ascending=False).unique()
    month_options = [pd.to_datetime(d).strftime('%Y년 %m월') for d in available_dates]
    max_index = len(month_options) - 1
    temp_data_row = df_cleaned.iloc[0]
    shop_name = temp_data_row['MCT_NM']
    
    st.subheader('👀 가맹점 개요')
    
    with st.container(border=True):
        col_1, col_2, col_3, col_4 = st.columns(4)
        
        with col_1:
            st.markdown(f"**{MAPPING_DCT.get('MCT_NM')}**")
            st.markdown(temp_data_row.get('MCT_NM', '정보 없음'))            
        
        with col_2:
            st.markdown(f"**{MAPPING_DCT.get('MCT_BSE_AR')}**")
            st.markdown(temp_data_row.get('MCT_BSE_AR', '정보 없음'))
                    
        with col_3:
            st.markdown(f"**{MAPPING_DCT.get('HPSN_MCT_ZCD_NM')}**")
            st.markdown(temp_data_row.get('HPSN_MCT_ZCD_NM', '정보 없음'))
                    
        with col_4:
            st.markdown(f"**{MAPPING_DCT.get('HPSN_MCT_BZN_CD_NM')}**")
            st.markdown(temp_data_row.get('HPSN_MCT_BZN_CD_NM', '정보 없음'))
    
    
    
    
    
    st.write('')
    
    
    
    
    
    col_left, col_center, col_right = st.columns([1, 4, 1])

    with col_left:
        disable_left = st.session_state.current_month_index == max_index
        if st.button('◀ 이전', disabled=disable_left, use_container_width=True):
            if st.session_state.current_month_index < max_index:
                st.session_state.current_month_index += 1
                st.rerun()

    with col_center:
        current_index = st.session_state.current_month_index
        selected_month_str = month_options[current_index]
        st.markdown(
            f"<h4 style='text-align: center; margin: 0;'>{selected_month_str}</h4>", 
            unsafe_allow_html=True
        )
        
    with col_right:
        disable_right = st.session_state.current_month_index == 0
        if st.button('다음 ▶', disabled=disable_right, use_container_width=True):
            if st.session_state.current_month_index > 0:
                st.session_state.current_month_index -= 1
                st.rerun()
    
    selected_date = pd.to_datetime(selected_month_str, format='%Y년 %m월')
    
    X_predict = X_processed[df_cleaned['TA_YM'] == selected_date]
    df_cleaned_selected = df_cleaned[df_cleaned['TA_YM'] == selected_date]
    
    if X_predict.empty:
        st.error(f'선택된 월에 해당하는 데이터가 전처리 후 존재하지 않아요.')
        st.stop()
    
    model_features = LGBM_MODEL.feature_name_
    
    X_final = pd.DataFrame(0, index=X_predict.index, columns=model_features)
    for col in X_final.columns:
        if col in X_predict.columns:
            X_final.loc[X_final.index, col] = X_predict.loc[X_predict.index, col]
                
    final_data_row = df_cleaned_selected.iloc[0]
    latest_month = selected_month_str
    
    with st.spinner(''):
        prediction, proba, top_features_df, chart_data = generate_shap_report(
            LGBM_MODEL, X_final, model_features
        )
    
    
    
    
    
    st.write('')





    col1, col2, col3, col4, col5 = st.columns(5)

    with col3:
        if prediction == 1:
            st.image('https://media.giphy.com/media/IzcFv6WJ4310bDeGjo/giphy.gif')
        else:
            st.image('https://media.giphy.com/media/ZdNlmHHr7czumQPvNE/giphy.gif')




            
    st.write('')





    if prediction == 1:
        st.error(f'### 🚨 즉각적인 대응이 필요해요. ({proba * 100:.2f}점)')
    else:
        st.success(f'### ✨ 안정 상태예요. ({proba * 100:.2f}점)')





    st.write('')





    st.subheader(f'🔮 위기 예측 기여도 (상위 5개 변수)')
    
    st.info(f'😊 SHAP 값은 특정 데이터 포인트의 예측 결과가 나오기까지 각 변수가 얼마나 기여했는지 설명하는 지역적 해석 지표예요.')

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            y=alt.Y('Feature_KOR', 
                    title='변수',
                    sort='-x'
            ),
            x=alt.X('SHAP_Value', 
                    title='SHAP 값'
            ),
            color=alt.condition(
                alt.datum.SHAP_Value > 0,
                alt.value(PASTEL_RED),
                alt.value(PASTEL_BLUE)
            ),
            tooltip=[
                alt.Tooltip('Feature_KOR', title='변수'),
                alt.Tooltip('SHAP_Value', title='SHAP 값', format='.4f')
            ]
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
    
    with st.container(border=True):
        legend_content = f"""
            <div>
                <span style='color:{PASTEL_RED}; font-weight:bold;'>■ 위기 신호</span>: 해당 변수의 값이 위기 확률을 높이는 방향으로 기여했어요.
                <br>
                <span style='color:{PASTEL_BLUE}; font-weight:bold;'>■ 안정 신호</span>: 해당 변수의 값이 위기 확률을 낮추는 방향으로 기여했어요.
                <hr style="margin: 10px 0;">
                <p style='font-size: 13px; color: #666;'>SHAP 값은 0을 기준으로 하며, 막대가 길수록 예측 결과에 미치는 기여도가 커요.</p>
            </div>
        """
        st.markdown(legend_content, unsafe_allow_html=True)





    st.text('')
    
    
    
    
    
    st.subheader('🗝️ 변수 중요도 (상위 10개 변수)')

    st.info("😊 이 차트는 특정 시점의 기여도(SHAP)가 아닌, 모델이 학습 과정에서 전반적으로 가장 중요하게 사용한 변수를 보여줘요.")

    importance_chart = plot_feature_importances(LGBM_MODEL, MAPPING_DCT)
    st.altair_chart(importance_chart, use_container_width=True)





    st.write('')





    st.subheader(f'📊 주요 변수 월별 추세')

    st.info('☺️ 0%에 가까울수록 상위예요.')

    plot_columns = ['TA_YM'] + KEY_VARIABLES
    df_plot = df_cleaned[plot_columns].copy()

    df_long = df_plot.melt(
        id_vars=['TA_YM'], 
        value_vars=KEY_VARIABLES, 
        var_name='Feature_ENG', 
        value_name='Value'
    )
    df_long['Feature_KOR'] = df_long['Feature_ENG'].apply(lambda x: map_feature_name(x, MAPPING_DCT))

    midpoint_map = {
        '10% 이하': 0.05,
        '10-25%': 0.175,
        '25-50%': 0.375,
        '50-75%': 0.625,
        '75-90%': 0.825,
        '90% 초과': 0.95
    }
    reverse_midpoint_map = {v: k for k, v in midpoint_map.items()}

    df_long['Value_Rank'] = df_long['Value'].map(reverse_midpoint_map)

    df_long['Value_Rank'] = pd.Categorical(
        df_long['Value_Rank'], 
        categories=midpoint_map.keys(),
        ordered=True
    )
    
    current_categories = df_long['Value_Rank'].cat.categories.tolist()

    filter_date_str = selected_date.isoformat()

    variable_selection = alt.selection_point(
        fields=['Feature_KOR'], 
        nearest=True,
        on='mouseover', 
        empty='none'
    )

    base = alt.Chart(df_long).encode(
        x=alt.X('TA_YM', title='연월', axis=alt.Axis(format='%Y-%m')),
        y=alt.Y(
            'Value_Rank',
            title='구간'
        ),
        color=alt.Color('Feature_KOR', title='주요 변수'),
        tooltip=[
            alt.Tooltip('TA_YM', title='연월', format='%Y-%m'), 
            alt.Tooltip('Feature_KOR', title='변수'),
            alt.Tooltip('Value_Rank', title='구간')
        ],
    )

    hover_points = base.mark_point().encode(
        opacity=alt.value(0),
        size=alt.value(50)
    ).add_params(
        variable_selection
    )

    visual_line_layer = base.mark_line().encode(
        strokeWidth=alt.condition(
            variable_selection, 
            alt.value(10),
            alt.value(5)
        ),
        opacity=alt.condition(
            variable_selection, 
            alt.value(1.0),
            alt.value(0.3)
        ),
    )

    interactive_group = hover_points + visual_line_layer
    
    final_combined_chart = (interactive_group).interactive()
    
    st.altair_chart(final_combined_chart, use_container_width=True)