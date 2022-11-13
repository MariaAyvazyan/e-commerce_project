## Промежуточный проект 
### Вариант 1: e-commerce 
# Импортируем необходимые библиотеки

import pandas as pd
import seaborn as sns
import numpy as np
from operator import attrgetter

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
#Считаем таблицу с данными о пользователях 

customers_df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-m-ajvazjan-23/olist_customers_dataset.csv')
customers_df.head()
customer_id — позаказный идентификатор пользователя

customer_unique_id —  уникальный идентификатор пользователя  (аналог номера паспорта)

customer_zip_code_prefix —  почтовый индекс пользователя

customer_city —  город доставки пользователя

customer_state —  штат доставки пользователя


customers_df.customer_id.nunique()
customers_df.customer_unique_id.nunique()
Количество уникальных значений cutomer_id больше, чем уникальных значений customer_unique_id. Это может означать, что одному и тому же пользователю присвается разный customer_id каждый раз при совершении покупки. 
#Считаем таблицу с данными о заказах

orders_df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-m-ajvazjan-23/olist_orders_dataset.csv')
orders_df.head()
order_id —  уникальный идентификатор заказа (номер чека)

customer_id —  позаказный идентификатор пользователя

order_status —  статус заказа

order_purchase_timestamp —  время создания заказа

order_approved_at —  время подтверждения оплаты заказа

order_delivered_carrier_date —  время передачи заказа в логистическую службу

order_delivered_customer_date —  время доставки заказа

order_estimated_delivery_date —  обещанная дата доставки


#Считаем таблицу с данными о товарных позициях 

items_df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-m-ajvazjan-23/olist_order_items_dataset.csv')
items_df.head()
order_id —  уникальный идентификатор заказа (номер чека)

order_item_id —  идентификатор товара внутри одного заказа

product_id —  ид товара (аналог штрихкода)

seller_id — ид производителя товара

shipping_limit_date —  максимальная дата доставки продавцом для передачи заказа партнеру по логистике

price —  цена за единицу товара

freight_value —  вес товара


### 1. Сколько у нас пользователей, которые совершили покупку только один раз? 
Для целей выполнения данного проекта покупкой будет считаться заказ, имеющий статус "delivered" и имеющий дату доставки в столбце order_delivered_customer_date. 

Для ответа на вопрос объединим датафреймы oredres_df и customers_df по customer_id. 
orders_merged = orders_df.merge(customers_df, on = "customer_id", how = "inner")
orders_merged
#посчитаем уникальных пользователей с одной покупкой 

single_order = orders_merged[(orders_merged.order_status == 'delivered')&(orders_merged.order_delivered_customer_date.notnull())]\
             .groupby('customer_unique_id', as_index=False)\
             .agg({'order_id':'nunique'})\
             .query('order_id == 1').count().order_id
single_order
Итого: 90549 пользователей совершили только одну покупку. 
#Посчитаем процент от общего числа покупателей. 
round((single_order/orders_merged.customer_unique_id.nunique()*100),2)
Более 94% наших пользователей совершили покупку всего один раз. Необходимо обратить на это внимание и разработать возможные способы удержания клиентов.  
### 2. Сколько заказов в месяц в среднем не доставляется по разным причинам (вывести детализацию по причинам)? 
В качестве недоставленных товаров рассмотрим те товары, которые имеют статус, отличный от 'delivered'. 
Но для начала убедимся, что все заказы со статусом delivered также имеют order_delivered_customer_date. 
orders_merged.dtypes
#Приведем данные к нужному формату

orders_merged['order_purchase_timestamp']= pd.to_datetime(orders_merged['order_purchase_timestamp'])
orders_merged['order_approved_at']= pd.to_datetime(orders_merged['order_approved_at'])
orders_merged['order_delivered_carrier_date']= pd.to_datetime(orders_merged['order_delivered_carrier_date'])
orders_merged['order_delivered_customer_date']= pd.to_datetime(orders_merged['order_delivered_customer_date'])
orders_merged['order_estimated_delivery_date']= pd.to_datetime(orders_merged['order_estimated_delivery_date'])
orders_merged.dtypes
#Посмотрим, есть ли заказы, которые имеют статус delivered, но не имеют даты доставки, и если да, то сколько таких случаев

orders_merged[(orders_merged.order_status == 'delivered')&(orders_merged.order_delivered_customer_date.isna())].count()

Из получившихся данных видно, что 8 заказов, имеющих статус delivered, не имеют даты доставки в столбце order_delivered_customer_date, а один заказ, имеющий статус delivered, и вовсе не был передан курьеру (order_delivered_carrier_date).
#Отфильтруем заказы, не имеющие даты доставки, и посмотрим, какие статусы вообще есть в датасете

not_delivered_orders = orders_merged[orders_merged.order_delivered_customer_date.isna()]
#Выведем все существующие статусы заказов
not_delivered_orders.order_status.value_counts()
#проверим заказы со статусом processing
orders_merged[orders_merged.order_status == 'processing']
Исходя из данных, можем предположить, что с момента размещения заказа товар проходит несколько стадий: 

1) shipped - отправлен 
2) canceled - отменен 
3) unavailable - недоступен (возможно, нет в наличии) 
4) invoiced - выставлен счет 
5) processing - вероятнее всего, системный сбой или обработка платежа. Мы видим, что заказ подтвержден, но не выдан курьеру. 
6) delivered - доставлен. Хотя в данном случае дата доставки пропущена. 
7) created - создан 
8) approved - подтвержден

Из всех статусов в число недоставленных, на мой взгляд, можно включить не все. По логике работы интернет-магазинов, статусы created, approved, invoiced и shipped должны изменяться в ходе продвижения по логистической цепочке и в конечном итоге перейти в статус delivered. Но для того, чтобы быть уверенными, что мы можем не включать их в число недоставленных, посмотрим, какая для них была указана ожидаемая дата доставки и наступила ли эта дата. 
not_delivered_orders['order_purchase_by_month']= pd.to_datetime(orders_merged['order_purchase_timestamp']).dt.month
not_delivered_orders['order_estimated_delivery_by_month']= pd.to_datetime(orders_merged['order_estimated_delivery_date']).dt.month
#Проверим последнюю дату заказа, чтобы приблизительно понять максимальную дату, за которую у нас имеются данные

orders_merged.order_purchase_timestamp.max()
#Сравним данные ожидаемой даты доставки с полученной выше датой 

not_delivered_orders.query("order_estimated_delivery_date < '2018-10-17'")\
                    .groupby('order_status', as_index = False)\
                    .agg({'order_id':'count'})\
                    .sort_values(by='order_id', ascending = False)
Видим, что количество недоставленных заказов с интересующими нас статусами (created, approved, invoiced и shipped) не изменилось. Это означает, что заказы не были доставлены в срок, а значит, вряд ли поменяют свой статус, как предполагалось ранее. 
Следовательно, мы можем отнести данные заказы к категории недоставленных, наряду с отмененными (canceled) и недоступными (unavailable), а также наряду с теми, которые имеют статус delivered, но не имеют даты доставки. 

Посчитаем с учетом этого среднее количество недоставленных товаров в месяц. 
not_delivered_per_month = not_delivered_orders.groupby(['order_estimated_delivery_by_month','order_status'], as_index=False)\
                    .agg({'order_id':'count'})\
                    .sort_values(by='order_estimated_delivery_by_month')
                    
not_delivered_per_month.head(6)
#Посмотрим, сколько в среднем заказов не доставляется, сгруппировав по причинам

not_delivered_per_month.pivot_table(index = 'order_status', values = 'order_id', aggfunc = 'mean')
#Среднее количество недоставленных заказов в месяц всего: 

not_delivered_per_month.order_id.mean().round(1)
Видим, что достаточно большое количество заказов не доставляется по причине недоступности товара. Причем предполагаю, что товар числится на сайте магазина и отмена происходит уже в процессе оформления заказа. Это говорит о том, что необходимо наладить работу сайта и, как минимум, не выставлять на полки недоступные к заказу товары.

Также очень большое количество заказов висит в статусе shipped (отправлены), но сроки ожидаемой доставки уже давно прошли. Учитывая, что заказ не отменен, тут может быть два варианта: 
1. Заказ на самом деле отменет, но есть внутренний сбой в админке компании, из-за которого не меняются статусы. В этом случае можно предположить, что товар был утерян по дороге. 
2. Товар застрял где-то на границе, если осуществляется международная перевозка. 

Много отмененных заказов. Примем отмененных как покупателями, так и продавцами. Вероятная причина - утеря товара со стороны продавка или слишком долгое ожидание доставки со стороны покупателя. 

Дальшейшего изучения требует также статус processing. На данном этапе у нас недостаточно данных для этого, но необходимо изучить, что означает этот статс - снутренний сбой при оформлении заказа или обработка платежа/заказа, или что-то иное. 

Учитывая временной отрезок, за которые у нас есть данные, предположу, что компания работает недавно и не все процессы доставки налажены. Это все гипотезы, для подтверждения которых необходим дальнейший анализ. 
### 3. По каждому товару определить, в какой день недели товар чаще всего покупается.
#Для начала объединим датафреймы для получение полной информации о заказах

final_df = orders_merged.merge(items_df, on = 'order_id', how = 'left')

final_df.dtypes
#Приведем дату покупки к формату дня недели 

final_df['weekdays'] = final_df.order_purchase_timestamp.dt.day_name()
Выше было обозначено, что в данном проекте покупкой будут считать только те заказы, которые имеют статус delivered и имеющий дату доставки до покупателя. Отфильтруем данные, исходя из этого. 
final_purchase_df = final_df[(final_df.order_status == 'delivered')&(final_df.order_delivered_customer_date.notnull())]
#Сгруппируем данные по дням недели

top_sellers = final_purchase_df.groupby(['weekdays', 'product_id'], as_index = False)\
                               .agg({'order_id':'count'}) \
                               .drop_duplicates(subset=["product_id"], keep='first')\
                               .sort_values(['order_id', 'product_id'], ascending = False)\
                               .rename(columns={'order_id': 'amount'})
                              
top_sellers.groupby('product_id', as_index = False).agg({'amount':'max'}).sort_values(['product_id'],ascending=False)
top_sellers.head()
### 4. Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)? Не стоит забывать, что внутри месяца может быть не целое количество недель. Например, в ноябре 2021 года 4,28 недели. И внутри метрики это нужно учесть. 
#Рассчитаем количество недель в каждом месяце

final_purchase_df['weeks'] = final_purchase_df.order_purchase_timestamp.dt.days_in_month / 7
final_purchase_df['purchase_month'] = final_purchase_df.order_purchase_timestamp.dt.to_period('M') 
avg_order = final_purchase_df.groupby(['customer_unique_id','purchase_month','weeks'], as_index=False)\
                             .agg({'order_id':'nunique'})
avg_order
#Разделим среднее количество покупок в месяц на количество недель в месяце

avg_order.order_id = avg_order.order_id / avg_order.weeks
Среднее количество заказов в неделю у каждого из пользователей:
avg_order
#Среднее количество покупок на одного пользователя по месяцам

avg_order.groupby('purchase_month')['order_id'].median().plot();
Так как среднее по средним считать нельзя, то я посчитала медиану частотности заказов помесячно и построила базовый график для лучшей визуализации.  

Из графика можем увидеть сильные повышения частотности по февралям и менее сильные по месяцам, в которых 30 дней.
### 5. Используя pandas, проведи когортный анализ пользователей. В период с января по декабрь выяви когорту с самым высоким retention на 3й месяц.

Деление на когорты будем производить по дате первого заказа. 
#Для начала подготовим датафрейм и оставим в нем только нужные столбцы

cohort_df = final_purchase_df.groupby(['customer_unique_id','purchase_month'], as_index = False )\
                            .agg({'order_id':'count'})\
                            .sort_values(by = 'purchase_month')\
                            .rename(columns = {'order_id': 'purchase amount'})
cohort_df.dtypes
cohort_df.purchase_month.min()
cohort_df.purchase_month.max()
cohort_df
#Добавим датафрейм с датой первого заказа для каждого отдельного пользователя

first_purchase = cohort_df.groupby('customer_unique_id', as_index = False)\
                           .agg({'purchase_month':'min'})\
                           .rename(columns = {'purchase_month': 'first_purchase'})
first_purchase
#Объединим оба датафрейма

cohort_df = cohort_df.merge(first_purchase, how = 'inner', on = 'customer_unique_id')
cohort_df
cohort_df.loc[cohort_df['customer_unique_id'] == '830d5b7aaa3b6f1e9ad63703bec97d23']
#Сгруппируем пользователей по дате первой и "последней" покупки 
cohort_df = cohort_df.groupby(['first_purchase','purchase_month'], as_index=False) \
                     .agg({'customer_unique_id':'count'})
cohort_df
#Разобьем пользователей на когорты, вычислив разницу между месяцем покупки и первым месяцем покупки 

cohort_df['cohort'] = (cohort_df.purchase_month - cohort_df.first_purchase)\
                       .apply(attrgetter('n'))
cohort_df
cohort_df_pivot = cohort_df.pivot_table(index='first_purchase', columns='cohort', values='customer_unique_id')

cohort_df_pivot
#Посчитаем retention rate - какое количество пользователей из нашей когорты сделало покупку в интересующий нас период. 

retention = cohort_df_pivot.divide(cohort_df_pivot.iloc[:, 0], axis=0)
retention
retention[3].max().round(4)
Максимальный retention rate на третий месяц после совершения первого заказа среди пользователей, совершивших покупку, приходится на 2017-06 и составляет примерно 0.43%

Визуализируем для наглядности. 
plt.figure(figsize=(20,8))
plt.title('Retention rate')
sns.heatmap(data = retention, annot=True, cmap='Oranges', mask=retention.isnull(), vmin=0.0,vmax=0.009, fmt='.2%');
### 6. Часто для качественного анализа аудитории использую подходы, основанные на сегментации. Используя python, построй RFM-сегментацию пользователей, чтобы качественно оценить свою аудиторию.
Для сегментации создадим следующие метрики:
R (recency) - время от последней покупки пользователя до текущей даты, F (frequency) - суммарное количество покупок у пользователя за всё время, M (monetary) - сумма покупок за всё время.

Так как мы определились считать покупкой заказы со статусом delivered и наличием даты доставки, будем использовать отфильтрованный ранее по этому критерию датафрейм final_purchase_df, но оставим в нем только необходимые столбцы. 
final_purchase_rfm = final_purchase_df[['customer_unique_id','order_id','order_purchase_timestamp','price']]
final_purchase_rfm.dtypes
Посчитаем метрику Recency, для этого найдем разницу во времени между последней покупкой пользователя и последней датой заказа (так как имеются не все данные) 
last_purchase = final_purchase_rfm\
                .groupby(['customer_unique_id'], as_index = False)\
                .agg({'order_purchase_timestamp':'max'})
last_purchase['recency'] = ((last_purchase.order_purchase_timestamp.max()-last_purchase.order_purchase_timestamp) / np.timedelta64(1, 'D')).astype(int)
last_purchase.recency.describe()
last_purchase.recency.hist()
Расчитаем метрику Frequency - частоту заказов на каждого пользователя за указанный период. 
frequency = final_purchase_rfm\
            .groupby(['customer_unique_id'], as_index = False)\
            .agg({'order_id':'nunique'})\
            .rename(columns = {'order_id':'frequency'})
frequency.frequency.describe()
Расчитаем метрику Monetary - сумму покупок каждого клиента за все время.
monetary = final_purchase_rfm\
            .groupby(['customer_unique_id'], as_index = False)\
            .agg({'price':'sum'})\
            .rename(columns = {'price':'monetary'})
#Объединим полученные метрики в один датафрейм 

rfm = last_purchase.merge(frequency, how = 'inner', on = 'customer_unique_id').merge(monetary, how = 'inner', on = 'customer_unique_id')
rfm = rfm.drop(columns = ['order_purchase_timestamp'])
rfm
Проведем сегментацию recency на основании квантилей и присвоим пользователям соответствующие ранги: 1 - пользователи, сделавшие заказ очень давно, 4 - сделавшие заказ недавно. 
quantile_recency = rfm.recency.quantile(q=[0.25,0.5,0.75]).round(-1)
quantile_recency

def recency_score(x):
    if x <= quantile_recency[0.25]:
        return 4
    elif x <= quantile_recency[0.5]:
        return 3
    elif x <= quantile_recency[0.75]:
        return 2
    else:
        return 1

rfm['recency_score'] = rfm.recency.apply(lambda x: recency_score(x)) 
Теперь внимательнее рассмотрим частоту заказов (frequency). В данном случае деление на сегменты по квантилям будет нецелесообразно, так как большая часть клиентов сделала не более одного заказа. 
rfm.query('frequency == 1')
rfm.query('frequency == 2')
rfm.query('frequency > 2 <= 5')
rfm.query('frequency > 5')
#На графике отчетливо видна разница в количестве покупаемых товаров

sns.distplot(rfm.frequency, kde = False);
Ориентируясь на полученные данные, присвоим пользователям ранги, где 1 - пользователи только лишь с 1 заказом, 4 - пользователи с более, чем 5 заказами. 
def frequency_score(x):
    if x == 1:
        return 1
    elif x == 2:
        return 2
    elif x < 2 <= 5:
        return 3
    else:
        return 4
    
rfm['frequency_score'] = rfm.frequency.apply(lambda x: frequency_score(x))
Проведем сегментацию monetary на основании квантилей и присвоим пользователям соответствующие ранги: 1 - пользователи, сделавшие заказ на минимальную сумму, 4 - пользователи с высоким чеком. 
rfm.monetary.describe()
quantile_monetary = rfm.monetary.quantile(q=[0.25,0.5,0.75]).round(-1)
quantile_monetary
def monetary_score(x):
    if x <= quantile_monetary[0.25]:
        return 1
    elif x <= quantile_monetary[0.5]:
        return 2
    elif x <= quantile_monetary[0.75]:
        return 3
    else:
        return 4
    
rfm['monetary_score'] = rfm.monetary.apply(lambda x: monetary_score(x))
rfm['RFM_score'] = rfm.recency_score.astype(str)\
                 + rfm.frequency_score.astype(str)\
                 + rfm.monetary_score.astype(str)
total_rfm = rfm.groupby('RFM_score', as_index = False)\
    .agg({'customer_unique_id':'count'})\
    .rename(columns = {'customer_unique_id': 'customer_count'})\
    .sort_values(by = 'customer_count', ascending = False)
Построим график для визуализации
plt.figure(figsize=(20,8))
sns.barplot(data = total_rfm , x='RFM_score', y= 'customer_count'); 
total_rfm['percentage'] = ((total_rfm.customer_count / total_rfm.customer_count.sum() * 100).round(2)).astype(str) + '%'
total_rfm
В результате получаем слишком большое количество кластеров, такой формат представляется неудобным для дальнейшего анализа. Попробуем объединить клиентов по общей сумме рангов, чтобы посмотреть на общую картину распределения клиентов.
rfm['RFM_total_score'] = rfm.recency_score\
                       + rfm.frequency_score\
                       + rfm.monetary_score
rfm_sum = rfm.groupby('RFM_total_score', as_index = False)\
    .agg({'customer_unique_id':'count'})\
    .rename(columns = {'customer_unique_id': 'customer_count'})\
    .sort_values(by = 'customer_count', ascending = False)
rfm_sum['percentage'] = ((rfm_sum.customer_count / rfm_sum.customer_count.sum() * 100).round(2)).astype(str) + '%'
rfm_sum
sns.barplot(data = rfm_sum , x='RFM_total_score', y= 'customer_count');
Метрика Recency расчитывалась как разница между последней покупкой клиента и последней датой, представленной в данных. Данные неполные, поэтому за расчет бралась именно дата последней покупки, представленная в таблице, а не текущая дата. Результаты и описание полученных данных показали, что мы можем использовать метод кластеризации по квантилям. В результате пользователи были разбиты на 4 группы, где 1 - это те, кто делал последний заказ очень давно (более 340 дней назад), а 4 - пользователи, вернувшиеся за покупками недавно (менее 110 дней назад).

Метрика Frequency вычислялась как общее количество заказов за весь период. В данном случае описание данных показало, что большая часть наших пользователей совершили всего одну покупку, а значит, кластеризация по квантилям нецелесообразна. Изучив данные немного детальнее, пришла к выводу, что можно разделить пользователей также на 4 группы, где 1 и 2 - пользователи, совершившие одну и две покупки соответственно, 3 - пользователи, сделавшие более 2 и до 5 заказов включительно, и 4 - пользователи, сделавшие заказ более 5 раз. 

Метрика Monetary рассчитывалась как общая сумма заказов за все время. Здесь я также использвоала кластеризацию посредством квантилей. В результате также получилось 4 ранга. 

Все полученные ранги я объединила в общий RFM score как отдельно по кажной метрике, так и в виде суммы рангов. 

В таблице с суммой рангов нас, прежде всего, интересуют клиенты с рангом 10 и выше, так как это клиенты, часто совершающие заказы с большой общей суммой покупок, с момента последнего заказа которых прошло немного времени. Мы видим, что общая доля таких клиентов не составляет и одного процента. 
Для того, чтобы решить, как взаимодействовать с такими клиентами, следует внимательнее ознакомиться с общим рангом и рассмотреть его в разрезе rfm score по кажой метрике - recency. frequency, monetary. Далее, изучить, какие показатели страдают больше всего. Также следует обратить внимание, возможно ли разделить таких клиентов на группы по другим критериям - возможно, по географическому или половому признаку. Все эти данные должны помочь для решения вопроса о грамотном продвижении и увеличении числа лояльных клиентов. 


