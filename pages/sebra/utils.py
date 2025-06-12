import sqlite3
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
import os
from typing import List, Tuple
from datetime import timedelta, datetime

con = sqlite3.connect('assets/sebra/sebra.db', check_same_thread=False)
read_sql = lambda table_name, index_col: pd.read_sql(sql=f'select * from {table_name}', con=con, index_col=index_col)
df_clients = read_sql('clients', 'CLIENT_ID')
df_orgs = read_sql('organizations', 'ORGANIZATION_ID')
df_primary_orgs = read_sql('primary_organizations', 'PRIMARY_ORG_CODE')
df_payments = read_sql('payments', 'PAYMENT_ID')
df_payments['SETTLEMENT_DATE'] = pd.to_datetime(df_payments['SETTLEMENT_DATE']).dt.round('D')
df_sebra = read_sql('sebra_codes', 'SEBRA_PAY_CODE')

pay_codes: List[int] = sorted(df_payments['SEBRA_PAY_CODE'].unique())
# PIES
pies = {
    'labels': np.sort(df_payments['SEBRA_PAY_CODE'].unique()),
    'to_remove': [18, 20, 80, 90] 
}
pies['other_label'] = 'other: ' + '+'.join(map(str, pies['to_remove']))
pies['labels'] = [label for label in pies['labels'] if label not in pies['to_remove']] + [pies['other_label']]

def compare_codes() -> go.Figure:
    per_code: pd.DataFrame = df_payments.groupby('SEBRA_PAY_CODE', as_index=False)['AMOUNT'].median().astype({'SEBRA_PAY_CODE': str})
    fig = px.bar(per_code, x='SEBRA_PAY_CODE', y='AMOUNT', log_y=True)
    fig.update_layout(title_text='Median Payment Amount per Pay Code')
    fig.update_yaxes(title_text='Median Payment Amount')
    fig.update_xaxes(title_text='SEBRA Pay Code')
    fig.update_traces(hovertemplate='Median Payment Amount: %{y}<extra></extra>')
    return fig

categories = {
    'Other': [
        'Българска национална телевизия',
        'Българска телеграфна агенция',
        'Българско национално радио',
        'Държавно предприятие Управление и стопанисване на язовири',
        'НФ - МВУ',
        'Предприятие за управление на дейностите по опазване на околната среда',
        'СТОЛИЧНА ОБЩИНА',
        'Сметна палата',
        'Съвет за електронни медии',
        'ЦБ - субсидии за общини'
    ],
    'Research': [
        'Българска Академия на Науките',
        'Държавно предприятие Научно-производствен център',
        'Национален статистически институт'
    ],
    'Security': [
        'ДКСИ',
        'Комисия за защита на конкуренцията',
        'Комисия за защита на личните данни',
        'Комисия за защита от дискриминация',
        'Комисия за публичен надзор над регистрираните одитори',
        'Комисия за регулиране на съобщеният',
        'Комисия за финансов надзор',
        'К-я за разкриване и обявяване на документи на ДС и РС на БНА',
        'КПКОНПИ',
        'Национално бюро за контрол на специалните разузнавателни средства',
        'ДА Държавен резерв и военновременни запаси'
    ],
    'Other Government': [
        'Централна избирателна комисия',
        'Омбудсман',
        'Народно събрание',
        'Администрация на президента',
        'Висш съдебен съвет'
    ],
    'Social': [
        'Централен бюджет - безлихвени заеми',
        'Учителски пенсионен фонд',
        'Фонд Гарантирани вземания на работници и служители към НОИ',
        'НОИ - ДОО',
        'НЗОК'
    ],
    'Energy': [
        'Агенция за ядрено регулиране',
        'Фонд Сигурност на електроенергийната система',
        'КОМИСИЯ ЗА ЕНЕРГ.И ВОДНО РЕГУЛИРАНЕ'
    ],
    'Agriculture': [
        'ДФ Земеделие-Разплащателна агенция',
        'ДЪРЖАВЕН ФОНД ЗЕМЕДЕЛИЕ'
    ],
    'Foreign': [
        'Национален фонд - Трансгранично сътрудничество',
        'НФ - ДРУГИ МЕЖДУНАРОДНИ ПРОГРАМИ',
        'МЗХ - ЧУЖДИ СРЕДСТВА',
        'МРРБ-чужди средства-Български ВиК холдинг ЕАД',
        'МТИТС - чужди средства - БДЖ-Пътнически превози ЕООД',
        'МТИТС-чужди средства-НК Железопътна инфраструктура',
        'НФ-средства от ЕС'
    ],
    'Education': [
        'Аграрен университет - Пловдив',
        'Академия за музикално  танцово и изобразително изкуство Проф. Асен Диамандиев - Пловдив',
        'Академия за музикално  танцово и изобразително изкуство Проф. Асен Диамандиев - Пловдив',
        'Бургаски държавен университет Проф. д-р Асен Златаров - Бургас',
        'Великотърновски университе Св. св. Кирил и Методий',
        'Висше военновъздушно училище Георги Бенковски - гр. Долна Митрополия',
        'Висше военноморско училище Н. Й. Вапцаров - Варна',
        'Висше строително училище Л. Каравелов',
        'Висше транспортно училище Т. Каблешков',
        'Висше училище по телекомуникации и пощи',
        'Военна академия Г. С. Раковски',
        'Икономически университет - Варна',
        'Лесотехнически университет',
        'Медицински университет',
        'Медицински университет - Плевен',
        'Медицински университет - Пловдив',
        'Медицински университет Проф. д-р Параскев Ив. Стоянов - Варна',
        'Минно-геоложки университет',
        'Национален военен университет Васил Левски - Велико Търново',
        'Национална академия за театрално и филмово изкуство',
        'Национална музикална академия П.Владигеров',
        'Национална спортна академия В. Левски',
        'Национална художествена академия',
        'Пловдивски университет Паисий Хилендарски',
        'Русенски университет Ангел Кънчев',
        'СУ Климент Охридски',
        'Селскостопанска академия',
        'Стопанска академия Димитър А. Ценов - Свищов',
        'Технически университет - Варна',
        'Технически университет - Габрово',
        'Технически университет - София',
        'Тракийски университет - Стара Загора',
        'УНСС',
        'Университет по архитектура  строите',
        'Университет по библиотекознание и информационни технологии',
        'Университет по хранителни технологии - Пловдив',
        'Химико-технологичен и металургичен',
        'Шуменски университет Епископ Константин Преславски',
        'Югозападен университет Неофит Рилски - Благоевград'
    ],
    'Ministries': [
        'Министерски съвет',
        'Министерство на външните работи',
        'Министерство на вътрешните работи',
        'Министерство на електронното управление',
        'Министерство на енергетиката',
        'Министерство на здравеопазването',
        'Министерство на земеделието и храните',
        'Министерство на икономиката и индустрията',
        'Министерство на иновациите и растежа',
        'Министерство на културата',
        'Министерство на младежта и спорта',
        'Министерство на образованието  мла',
        'Министерство на отбраната',
        'Министерство на правосъдието',
        'Министерство на регионалното развит',
        'Министерство на транспорта',
        'Министерство на труда и социалната',
        'Министерство на туризма',
        'Министерство на финансите',
        'Минстерство на околната среда и вод'
    ]
}

for category, primary_orgs in categories.items():
    df_primary_orgs.loc[df_primary_orgs['PRIMARY_ORGANIZATION'].isin(primary_orgs), 'Category'] = category

def make_pies() -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type':'pie'}, {'type':'pie'}]], 
        subplot_titles=['Total Amount', 'Total Payments'], 
        horizontal_spacing=.1
    )
    per_code = df_payments.groupby('SEBRA_PAY_CODE')['AMOUNT'].agg(['sum', 'size', 'median']).sort_index()
    per_code.loc[pies['other_label']] = per_code.loc[pies['to_remove']].sum()
    per_code = per_code.drop(pies['to_remove'])

    fig.add_traces(
        data=[
            go.Pie(
                labels=pies['labels'], values=per_code['sum'].round(),
                name='Total Amount', hovertemplate='Code: %{label}<br>Total Amount: %{value}<extra></extra>',
            ),
            go.Pie(
                labels=pies['labels'], values=per_code['size'], 
                name='Total Payments', hovertemplate='Code: %{label}<br>Total Payments: %{value}<extra></extra>',
            )
        ],
        rows=1, cols=[1, 2]
    )
    
    fig.update_layout(
        legend_traceorder='normal',
        legend_title='SEBRA Code',
        title_text='Payments for different SEBRA categories'
    )
    return fig

def make_time_series(hide_weekends: bool = False, log_scale: bool = False) -> go.Figure:
    per_day = df_payments.groupby('SETTLEMENT_DATE')['AMOUNT'].agg(['sum', 'size'])
    if hide_weekends: per_day = per_day.asfreq('D')
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_traces(
        data=[
            go.Scatter(
                x=per_day.index, y=per_day['sum'], 
                name='Total Amount', mode='lines+markers', 
                hovertemplate='Total Amount: %{y}<extra></extra>'
            ),
            go.Scatter(
                x=per_day.index, y=per_day['size'], 
                name='Total Payments', mode='lines+markers', 
                hovertemplate='Total Payments: %{y}<extra></extra>'
            )
        ],
        secondary_ys=[False, True]
    )
    
    fig.update_layout(
        title_text='Payments over Q1',
        xaxis_range=['2025-01-01', '2025-04-01'],
        hoversubplots="axis",
        hovermode="x"
    )
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Total Amount', secondary_y=False)
    fig.update_yaxes(title_text='Total Payments', secondary_y=True)
    if log_scale: fig.update_yaxes(type='log')
    return fig

def plot_primary_orgs() -> go.Figure:
    per_primary_org = (
        df_payments
        .merge(df_orgs, on='ORGANIZATION_ID')
        .merge(df_primary_orgs, on='PRIMARY_ORG_CODE')
        .groupby('PRIMARY_ORGANIZATION')
        ['AMOUNT']
        .agg(['median', 'size', 'sum'])
        .round()
    )
    per_primary_org['median'] = np.log10(per_primary_org['median'])
    fig = go.Figure()
    colorbar = {
        'title': {'text': 'Median Payment Amount'},
        'tickvals': [3.7, 4.7, 5.7, per_primary_org['median'].max()],
        'ticktext': ['5k', '50k', '500k', '']
    }
    marker = {
        'color': per_primary_org['median'],
        'colorbar': colorbar,
        'cmin': 3.7,
        'cmax': per_primary_org['median'].max(),
        'size': 5
    }

    model = sm.OLS(np.log10(per_primary_org['size']), sm.add_constant(np.log10(per_primary_org['sum']))).fit()
    xs = [per_primary_org['sum'].min(), per_primary_org['sum'].max()]
    ys = 10 ** model.predict(sm.add_constant(np.log10(xs)))
    
    # text = f'$R^2={model.rsquared:.3f}$<br>$\log_{{10}}(y)={model.params["const"]:.3f}+{model.params["sum"]:.3f}x$'
    hovertemplate_trendline = f'R-squared: {model.rsquared:.2f}<br>Slope: {model.params["sum"]:.2f}<br>Intercept: {model.params["const"]:.2f}<extra></extra>'
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line_color='orange', hovertemplate=hovertemplate_trendline))

    hovertemplate_scatter = '<b>%{customdata[0]}</b><br>Total Amount: %{x:,}<br>Total Payments: %{y}<extra></extra>'
    fig.add_trace(go.Scatter(x=per_primary_org['sum'], y=per_primary_org['size'], marker=marker, mode='markers', customdata=per_primary_org.index.values[:, np.newaxis], hovertemplate=hovertemplate_scatter))

    fig.update_layout(title_text='Payments per Primary Organization', showlegend=False)
    fig.update_xaxes(type='log', title_text='Total Payment Amount')
    fig.update_yaxes(type='log', title_text='Total Number of Payments')
    return fig

def compare_weekdays():
    df_with_weekday = df_payments.assign(weekday=df_payments['SETTLEMENT_DATE'].dt.day_name())
    amount_per_weekday = df_with_weekday.groupby('weekday')['AMOUNT'].median()
    payments_per_weekday = df_with_weekday.groupby(['SETTLEMENT_DATE', 'weekday']).size().groupby('weekday').median()
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    fig.add_traces(
        data=[
            go.Bar(
                x=weekdays, y=amount_per_weekday.reindex(weekdays), offsetgroup=0, name='Median Payment Amount',
                hovertemplate='Median Payment Amount: %{y}<extra></extra>'
            ),
            go.Bar(
                x=weekdays, y=payments_per_weekday.reindex(weekdays), offsetgroup=1, name='Median Amount of Payments',
                hovertemplate='Median Number of Payments: %{y}<extra></extra>'
            )
        ],
        secondary_ys=[False, True]
    )
    fig.update_layout(xaxis_title='Day of the Week', title_text='Payments per Weekday', barmode='group')
    fig.update_yaxes(title_text='Median Payment Amount', secondary_y=False)
    fig.update_yaxes(title_text='Median Amount of Payments', secondary_y=True)
    return fig

def plot_treemap() -> go.Figure:
    df = (
        df_payments
        .merge(df_orgs, on='ORGANIZATION_ID')
        .merge(df_primary_orgs, on='PRIMARY_ORG_CODE')
        .groupby('PRIMARY_ORGANIZATION', as_index=False)
        .agg({'Category': 'first', 'AMOUNT': 'median'})
    )
    df.columns = ['PRIMARY_ORGANIZATION', 'Category', 'Median Amount']
    fig = px.treemap(df, path=['Category', 'PRIMARY_ORGANIZATION'], values='Median Amount')
    fig.update_layout(autosize=False, width=1500, height=1500)
    return fig
