##
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


def convert_to_int(val):
    new_val = str(val)
    if new_val == ' ':
        new_val = 0
        return int(new_val)
    else:
        return int(new_val)


def convert_to_date(val):
    new_val = str(val)
    if new_val == ' ':
        new_val = ''
        return new_val
    else:
        new_val = dt.datetime.strptime(str(val), '%Y-%m-%d')
    return new_val


##
pd_item_data = pd.read_csv('../CaseData_2020_ID029/input_files/ItemData.txt', delimiter='|')

dtypes1_item_data = pd_item_data.dtypes

# unique items in 'items data'
pd_item_data['ITEM'] = pd_item_data['ITEM'].apply(str)
pd_item_data['concat'] = pd_item_data['ITEM'] + pd_item_data['CATEGORY'] + pd_item_data['COLOR']

# creating unique df by dropping duplicates
pd_item_data_uni = pd_item_data.drop_duplicates(subset='concat', keep='first').copy()

pd_item_data_uni['ITEM'] = pd_item_data_uni['ITEM'].apply(lambda i: int(i))

pd_item_data_uni.replace(' ', np.nan, inplace=True)

nan_row_in_item_data_uni = pd_item_data_uni.isnull().sum(axis=0)

print('size of items_df :', pd_item_data.shape)
print('size of unique_items_df :', pd_item_data_uni.shape)

dtypes2_items_data = pd_item_data_uni.dtypes

##
pd_customer_data = pd.read_csv('../CaseData_2020_ID029/input_files/CustomerData.txt', delimiter='|')
dtypes1_customer_data = pd_customer_data.dtypes

pd_customer_data['DOB'] = pd.to_datetime(pd_customer_data['DOB'], format='%Y-%m-%d')

pd_customer_data.replace(' ', np.nan, inplace=True)
dtypes2_customer_data = pd_customer_data.dtypes

nan_row_in_customer_data = pd_customer_data.isnull().sum(axis=0)

##
pd_trans_data = pd.read_csv('../CaseData_2020_ID029/input_files/TransactionsData.txt', delimiter='|')

dtypes1_trans_data = pd_trans_data.dtypes

# generating unique key on (USERID n TRACKNO)
pd_trans_data['USERID'] = pd_trans_data['USERID'].apply(str)
pd_trans_data['TRACKNO'] = pd_trans_data['TRACKNO'].apply(str)
pd_trans_data['concat_key'] = pd_trans_data['USERID'] + pd_trans_data['TRACKNO']
print('actual trans_data size :- ', pd_trans_data.shape)

# creating df with only duplicate values
pd_trans_data_dup = pd_trans_data[pd_trans_data.duplicated(['concat_key'])].copy()
print('Size od dup_trans_data :- ', pd_trans_data_dup.shape)

# creating unique df by dropping duplicate rows; it has blanks.
pd_trans_data_uni = pd_trans_data.drop_duplicates(subset='concat_key', keep='first').copy()
print('uni_trans_data size before dropna after drop_dup :- ', pd_trans_data_uni.shape)
dtypes2_trans_data = pd_trans_data_uni.dtypes

# converting columns to appropriate data_types before filling nan
pd_trans_data_uni['DELIVERYDATE'] = pd_trans_data_uni['DELIVERYDATE'].apply(convert_to_date)
pd_trans_data_uni['USERID'] = pd_trans_data_uni['USERID'].apply(lambda u: int(u) if u != ' ' else int(0))
pd_trans_data_uni['DISCOUNT'] = pd_trans_data_uni['DISCOUNT'].apply(lambda d: int(d) if d != ' ' else int(0))
pd_trans_data_uni['QTY'] = pd_trans_data_uni['QTY'].apply(lambda q: int(q) if q != ' ' else int(1))
pd_trans_data_uni['RATING'] = pd_trans_data_uni['RATING'].apply(lambda r: int(r) if r != ' ' else int(0))
pd_trans_data_uni['SHIPDAYS'] = pd_trans_data_uni['SHIPDAYS'].apply(lambda s: int(s) if s != ' ' else int(0))
pd_trans_data_uni['TRACKNO'] = pd_trans_data_uni['TRACKNO'].apply(lambda t: int(float(t)) if t != ' ' else int(0))

# replacing blanks with nan in 'unique df'
pd_trans_data_uni.replace(' ', np.nan, inplace=True)

# finding count of NaNs in df
nan_row_in_trans_data = pd_trans_data_uni.isnull().sum(axis=0)
nan_col = pd_trans_data_uni.isnull().sum(axis=1)

dtypes3_trans_data = pd_trans_data_uni.dtypes

##
# merging two dfs to find anomalies
df_trans = pd_trans_data_uni[pd_trans_data_uni['USERID'] != 0]
print('size of transaction & unique transaction:', df_trans.shape, pd_trans_data_uni.shape)
master_df = pd.merge(left=df_trans, right=pd_customer_data,
                     how='left', left_on='USERID', right_on='USERID')
print('master df size:', master_df.shape)

nan_row_in_master_df = master_df.isnull().sum(axis=0)
nan_col_in_master_df = master_df.isnull().sum(axis=1)

dtypes1_master_df = master_df.dtypes

##
# master = master_df[['ITEM', 'USERID', 'WEBBROWSER', 'PPC_ADD', 'PURCHASE', 'QTY', 'DISCOUNT', 'PAYMENT', 'WAREHOUSE',
#                   'SHIPDAYS', 'DELIVERYDATE', 'REVIEW', 'RATING', 'TRACKNO', 'TIMESTAMP', 'concat_key',
#                   'GENDER', 'DOB', 'COUNTRY', 'EDUCATION', 'HOBBY']]

master_df_new = pd.merge(left=master_df, right=pd_item_data_uni,
                         how='left', left_on='ITEM', right_on='ITEM')

print('master size:', master_df.shape)
print('unique item df:', pd_item_data_uni.shape)

dtypes1_master_df_new = master_df_new.dtypes

master_df_new[['GENDER', 'PURCHASE', 'CATEGORY', 'COLOR', 'PAYMENT']] = master_df_new[['GENDER', 'PURCHASE',
                                                                                       'CATEGORY', 'COLOR',
                                                                                       'PAYMENT']].astype('category')
master_df_new['PURCHASE_CAT'] = master_df_new['PURCHASE'].cat.codes
master_df_new['GENDER_CAT'] = master_df_new['GENDER'].cat.codes
master_df_new['CATEGORY_CAT'] = master_df_new['CATEGORY'].cat.codes
master_df_new['COLOR_CAT'] = master_df_new['COLOR'].cat.codes
master_df_new['PAYMENT_CAT'] = master_df_new['PAYMENT'].cat.codes
master_df_new['PURCHASEPRICE'] = master_df_new['PURCHASEPRICE'].apply(lambda pp: float(pp))
master_df_new['DISCOUNT'] = master_df_new['DISCOUNT'].div(100)
master_df_new['actual_price'] = master_df_new['SALEPRICE'] * master_df_new['DISCOUNT']
master_df_new['actual_price'] = master_df_new['SALEPRICE'] - master_df_new['actual_price']
master_df_new['checkout_price'] = master_df_new['actual_price'] * master_df_new['QTY']
master_df_new['profit'] = np.where(master_df_new['actual_price'] > master_df_new['PURCHASEPRICE'], 'YES', 'NO')
print('new_master df:', master_df_new.shape)

dtypes2_master_df_new = master_df_new.dtypes

nan_row_in_master_df_new = master_df_new.isnull().sum(axis=0)
##
master_df_new.to_csv(r'..\CaseData_2020_ID029\input_files\master_data.csv')

##
#  Overall Sales by category
sales_by_category = master_df_new['CATEGORY'].value_counts().sort_values()
sales = master_df_new.groupby(['PURCHASE', 'CATEGORY']).size().reset_index(name='freq')
purchase_label = sales['PURCHASE'].tolist()

sns.set(style='darkgrid')
sns.barplot(sales_by_category.index, sales_by_category.values, alpha=0.9)
plt.title('Transactional distribution of listed items')
plt.xlabel('Category', fontsize=12)
plt.xticks(rotation=90)
plt.ylabel('Number of occurrences', fontsize=12)
plt.show()

##
# transactions successful/unsuccessful/not-accounted
successful_trans = sales.groupby(['PURCHASE'])[['freq']].sum().reset_index()
# unaccounted = len(master_df_new.index) - (successful_trans['freq'].sum())
unaccounted = master_df_new['PURCHASE'].isna().sum()
new_row = {'PURCHASE': 'UNACCOUNTED', 'freq': unaccounted}
successful_trans = successful_trans.append(new_row, ignore_index=True)
# successful_trans['percent'] = successful_trans['freq'].apply(lambda p: (p/len(master_df_new.index))*100)

sns.set(style='darkgrid')
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(aspect='equal')
ax1.set_title('IS PURCHASE SUCCESSFUL?', fontsize=25)
fig.subplots_adjust(wspace=0)
successful_trans.plot(kind='pie', y='freq', ax=ax1, autopct='%1.1f%%',
                      startangle=90, shadow=False, labels=successful_trans['PURCHASE'],
                      legend=False, fontsize=20)
plt.show()

##
# Item sales by country
sub_data = master_df_new[['ITEM', 'QTY', 'DISCOUNT', 'COUNTRY', 'CATEGORY', 'COLOR',
                         'PURCHASEPRICE', 'SALEPRICE', 'actual_price', 'checkout_price',
                          'profit']].copy()
sub_data.info()

# sub_data_groupby = sub_data[['COUNTRY', 'CATEGORY', 'COLOR', 'actual_price', 'QTY']].copy()

# sub_data_groupby = sub_data_groupby.groupby(['COUNTRY', 'CATEGORY', 'COLOR']).sum().reset_index()
##
a = sub_data['COLOR'].unique()


##
# Profitable items and loss items
