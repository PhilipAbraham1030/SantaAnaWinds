# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:24:08 2021

@author: raini
"""

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9

df19741983 = pd.read_csv('19741983rev.csv',parse_dates=True)
df19841993 = pd.read_csv('19841993rev.csv',parse_dates=True)
df19942003 = pd.read_csv('19942003rev.csv',parse_dates=True)
df20042013 = pd.read_csv('20042013rev.csv',parse_dates=True)
df20142021 = pd.read_csv('20142021rev.csv',parse_dates=True)

frames = [df19741983,df19841993,df19942003,df20042013,df20142021]
df19742021 = pd.concat(frames)


df = df19742021[['DATE','HourlyDryBulbTemperature','HourlyDewPointTemperature',
                 'HourlyWetBulbTemperature','HourlySeaLevelPressure',
                 'HourlyWindSpeed','HourlyWindDirection','HourlyVisibility','HourlyPrecipitation']]

# Write dataframe to a csv file
#df19742021.to_csv('df19742021.csv')

df['DATE'] = pd.to_datetime(df['DATE'])

df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month_name()    
df['DayY'] = df['DATE'].dt.day_of_year

df.set_index('DATE', inplace=True)
df.info()


df[['HourlyDryBulbTemperature','HourlyDewPointTemperature',
                 'HourlyWetBulbTemperature','HourlySeaLevelPressure',
                 'HourlyWindSpeed','HourlyWindDirection','HourlyVisibility','HourlyPrecipitation']] = df[['HourlyDryBulbTemperature',
                 'HourlyDewPointTemperature','HourlyWetBulbTemperature','HourlySeaLevelPressure',
                 'HourlyWindSpeed','HourlyWindDirection','HourlyVisibility','HourlyPrecipitation']].apply(pd.to_numeric, errors='coerce',axis=1)



# Missing Values in wind direction column
plt.figure(figsize=(14,6))
sns.heatmap(data=pd.DataFrame(df['HourlyWindDirection']).isnull(),cbar=False)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.xlabel('Missing HourlyWindDirection Data - Shown as Light Bars')
plt.show()
                                                                                                          
# Let's break this down closely by plotting the percentage of HourlyWindDirection values missing for each year.
lawd=pd.DataFrame(df['HourlyWindDirection'])

a=lawd.count(axis=1).resample('A').sum() # non-NA's by year
b=lawd.isnull().sum(axis=1).resample('A').sum() # NA's by year
c=pd.concat([a,b],axis=1) # axis=1 row direction to bind the data 
c.columns = ['non-na','na']
c['na%'] = round((c['na']/(c['non-na']+c['na']))*100)
c['na%'].plot(marker='.',linestyle='dashed',color='black', title='Percentage of Missing HourlyWindDirection Data per Year')
plt.ylabel('Missing Data %')
plt.show()
# looks like from 1995 to 2005, each year 50% of wind direction is missing

# Lets look look at wind direction data for 1997 for a an example of a year with lots of missing wind direction
df['HourlyWindDirection']['1997-07'].plot()
df['HourlyWindDirection']['1997-07'].isnull().mean()
# compare with say 1994
df['HourlyWindDirection']['1994-07'].plot()
df['HourlyWindDirection']['1994-07'].isnull().mean()


# Histogram of wind directions
df['HourlyWindDirection'].hist(bins=50)
# prevailing winds
df['HourlyWindDirection'][(df['HourlyWindDirection']>240) & (df['HourlyWindDirection']<300)].hist()

# df19742021['HourlyWindDirection'] = df19742021['HourlyWindDirection'].astype(str)
# df19742021['HourlyWindDirection'] = df19742021['HourlyWindDirection'].str.replace('.0', '', regex=False)



# create a list of our conditions for wind direction
conditions = [
    (df['HourlyWindDirection'] ==0.0),
    (df['HourlyWindDirection']==90.0),
    (df['HourlyWindDirection']==180.0),
    (df['HourlyWindDirection']==270.0),
    (df['HourlyWindDirection']==360.0),
    (df['HourlyWindDirection'] > 0.0) & (df['HourlyWindDirection'] < 90.0),
    (df['HourlyWindDirection'] > 90.0) & (df['HourlyWindDirection'] < 180.0),
    (df['HourlyWindDirection'] > 180.0) & (df['HourlyWindDirection'] < 270.0),
    (df['HourlyWindDirection'] > 270.0) & (df['HourlyWindDirection'] < 360.0)
    ]

# create a list of the values we want to assign for each condition
values = ['N', 'E', 'S', 'W', 'N', 'NE', 'SE', 'SW', 'NW']

# create a new column and use np.select to assign values to it using our lists as arguments
df['HourlyWindDirectionTier'] = np.select(conditions, values, default='Unknown')

df['HourlyWindDirectionTier'].value_counts()

p=df['HourlyWindDirectionTier'].hist(bins=40,color='orange');p.set_facecolor('black')

# Test if an index contains duplicate values
df.index.is_unique

# How many values in an index are duplicate
df.index.duplicated().sum()

# =============================================================================
# Drop rows with duplicate index values
# Using duplicated(), we can also remove values that are duplicates. 
# Using the following line of code, when multiple rows share the same index, 
# only the first one encountered will remain — following the same order in 
# which the DataFrame is ordered, from top to bottom. All the others 
# will be deleted.
# =============================================================================
df = df.loc[~df.index.duplicated(), :]

df.info()
df.head()
df.tail()
df.describe()
df.index
df.columns

# Write dataframe to a csv file
# df.to_csv('df.csv')

######################################################################################

#df = pd.read_csv('df.csv')



### Analysis of wind direction effect on other weather variables

sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlySeaLevelPressure", data=df)
sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlyDryBulbTemperature", data=df)
sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlyDewPointTemperature", data=df)
sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlyWetBulbTemperature", data=df)
sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlyWindSpeed", data=df[df['HourlyWindSpeed']<100])
#Violin plot - Takes too long
g = sns.catplot(x="HourlyWindDirectionTier", y="HourlyWindSpeed", kind="violin", inner=None, data=df[df['HourlyWindSpeed']<100])
sns.swarmplot(x="HourlyWindDirectionTier", y="HourlyWindSpeed", color="k", size=6, data=df[df['HourlyWindSpeed']<100], ax=g.ax)



# Density plots of NE and SW winds for 2019 and 2020.
dfwne = df[(df['HourlyWindDirectionTier']=='NE') | (df['HourlyWindDirectionTier']=='SW')].loc['2019':'2020']
p9.ggplot(dfwne) + p9.aes(x='HourlyDryBulbTemperature', fill='HourlyWindDirectionTier', title='2019-2020') + p9.geom_density(alpha=0.5)
p9.ggplot(dfwne) + p9.aes(x='HourlyDewPointTemperature', fill='HourlyWindDirectionTier') + p9.geom_density(alpha=0.5)
p9.ggplot(dfwne) + p9.aes(x='HourlySeaLevelPressure', fill='HourlyWindDirectionTier') + p9.geom_density(alpha=0.5)


# Is the temperature and pressure differences between NE and SW statistically significant?

# Pressure
dfSPNE = df[df['HourlyWindDirectionTier']=='NE'].loc[:,'HourlySeaLevelPressure']
dfSPSW = df[df['HourlyWindDirectionTier']=='SW'].loc[:,'HourlySeaLevelPressure']

# Normality assumption check
dfSPNE.hist()
dfSPSW.hist()

# equal variance assumption check
dfSPNE.std()
dfSPSW.std()

# Implementing a two-sample t-test
from scipy import stats
stats.ttest_ind(dfSPNE,dfSPSW, nan_policy='omit',equal_var=True)



# DryBulb Temperature
# Pressure
dfDBNE = df[df['HourlyWindDirectionTier']=='NE'].loc[:,'HourlyDryBulbTemperature']
dfDBSW = df[df['HourlyWindDirectionTier']=='SW'].loc[:,'HourlyDryBulbTemperature']

# Normality assumption check
dfDBNE.hist()
dfDBSW.hist()

# equal variance assumption check
dfDBNE.std()
dfDBSW.std()

# Implementing a two-sample t-test
from scipy import stats
stats.ttest_ind(dfDBNE,dfDBSW, nan_policy='omit',equal_var=True)


# DewPoint Temperature
# Pressure
dfDPNE = df[df['HourlyWindDirectionTier']=='NE'].loc[:,'HourlyDewPointTemperature']
dfDPSW = df[df['HourlyWindDirectionTier']=='SW'].loc[:,'HourlyDewPointTemperature']

# Normality assumption check
dfDPNE.hist()
dfDPSW.hist()

# equal variance assumption check
dfDPNE.std()
dfDPSW.std()

# Implementing a two-sample t-test
from scipy import stats
stats.ttest_ind(dfDPNE,dfDPSW, nan_policy='omit',equal_var=False)


# I am curious to know if dew-point is different between NE and SW winds for random samples of these winds
# =============================================================================
# Blocking
# We're going to have another look at the same data but, this time, we'll use 
# blocking to improve our approach. Like last time, you'll be using a two-sample 
# t-test on data. This time, however, 
# you will control for Month as a blocking factor, sampling equally from NE and 
# SW wind data. You will need to extract a random subset of data from 
# both winds to run your test.
# =============================================================================
seed=0000
weatvari = ['HourlyDewPointTemperature', 'HourlySeaLevelPressure']
# Create subset blocks
# Equal proportions of each wind(NE and SW) extracted from each of the twelve months

# Norteast Winds
NEjan = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "January")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEfeb = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "February")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEmar = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "March")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEapr = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "April")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEmay = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "May")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEjun = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "June")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEjul = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "July")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEaug = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "August")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEsep = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "September")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEoct = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "October")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEnov = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "November")].dropna(subset=weatvari).sample(n=800, random_state= seed)
NEdec = df[(df.HourlyWindDirectionTier == "NE") & (df.Month== "December")].dropna(subset=weatvari).sample(n=800, random_state= seed)

# Southwest Winds
SWjan = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "January")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWfeb = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "February")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWmar = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "March")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWapr = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "April")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWmay = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "May")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWjun = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "June")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWjul = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "July")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWaug = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "August")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWsep = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "September")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWoct = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "October")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWnov = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "November")].dropna(subset=weatvari).sample(n=800, random_state= seed)
SWdec = df[(df.HourlyWindDirectionTier == "SW") & (df.Month== "December")].dropna(subset=weatvari).sample(n=800, random_state= seed)

# Combine blocks
subsetNE = pd.concat([NEjan,NEfeb,NEmar,NEapr,NEmay,NEjun,NEjul,NEaug,NEsep,NEoct,NEnov,NEdec])
subsetSW = pd.concat([SWjan,SWfeb,SWmar,SWapr,SWmay,SWjun,SWjul,SWaug,SWsep,SWoct,SWnov,SWdec])

# Perform the two-sample t-test
print(stats.ttest_ind(subsetNE['HourlyDewPointTemperature'],subsetSW['HourlyDewPointTemperature'],equal_var=False))
# =============================================================================
# Your t-test is significant, with a p-value (7.238386857592895e-228) very much under 0.05. This blocked design 
# has resolved the issue you had in the last exercise. You can see how this type of 
# blocking approach can be a useful way to improve your experimental design when a 
# confounding variable is present.
# =============================================================================

p9.ggplot(subsetNE) + p9.aes(x=weatvari, fill='Month') + p9.geom_density(alpha=0.3)
p9.ggplot(subsetSW) + p9.aes(x=weatvari, fill='Month') + p9.geom_density(alpha=0.3)



#  visualize the probability distribution of NE and SW samples in a single plot.
# Plotting the KDE Plot
# 2D density + marginal distribution:
sns.set_style("darkgrid")
sns.jointplot(data=pd.concat([subsetNE,subsetSW]), x='HourlyDewPointTemperature', 
              y='HourlySeaLevelPressure', hue='HourlyWindDirectionTier', kind='kde',
              fill=True, joint_kws={'alpha': 0.4}, height=15)
plt.suptitle("Sammple from Population data",y=1.08) # y= some height>1
#plt.show()

#  plot in Scalable Vector Graphics format, which stores the actual lines and 
# shapes of the chart so that you can print it at any size - even a giant 
# poster - and it will look sharp. 

#plt.savefig('plots/DP-SP-WindDirect.svg')


# Two-way ANOVA
#  How does the Dewpoint temperature vary between SW and NE winds in LA and of different 
# months (January and July)? We will use a two-way ANOVA to check for the 
# presence of significant variation in the dewpoint temperature and sea level pressure. 
# A two-way ANOVA will allow you to see which of these two factors, Wind Direction and Month, 
# have a significant effect on Dewpoint temperature. 
# We will implement the ANOVA with interactive effects to also see if wind direction and
# month are interactive - dependent on the other.

# plot
dfs = pd.concat([subsetNE,subsetSW])
dfs = dfs[(dfs.Month=='January')|(dfs.Month=='July')]

dfs.HourlyWindDirectionTier.value_counts()

p9.ggplot(dfs) + p9.aes(x='HourlyWindDirectionTier', 
                        y='HourlyDewPointTemperature', 
                        fill='Month') + p9.geom_boxplot() + p9.labs(title='This is for random sample containing 3200 total of NE and SW winds \n for January and July months')


p9.ggplot(dfs) + p9.aes(x='Month', 
                        y='HourlyDewPointTemperature', 
                        fill='HourlyWindDirectionTier') + p9.geom_boxplot() + p9.labs(title='This is for random sample containing 3200 total of NE and SW winds \n for January and July months', fill='Wind Direction')


p9.ggplot(dfs) + p9.aes(x='HourlyWindDirectionTier', 
                        y='HourlySeaLevelPressure', 
                        fill='Month') + p9.geom_boxplot() + p9.labs(title='This is for random sample containing 3200 total of NE and SW winds \n for January and July months')

p9.ggplot(dfs) + p9.aes(x='Month', 
                        y='HourlySeaLevelPressure', 
                        fill='HourlyWindDirectionTier') + p9.geom_boxplot() + p9.labs(title='This is for random sample containing 3200 total of NE and SW winds \n for January and July months', fill='Wind Direction')





import statsmodels as sm

## Dew-Point Temperature
# Run the ANOVA
model = sm.api.formula.ols('HourlyDewPointTemperature ~ Month + HourlyWindDirectionTier + HourlyWindDirectionTier:Month', data = dfs).fit()

# Extract our table
aov_table = sm.api.stats.anova_lm(model, typ=2)

# Print the table
print(aov_table)
# =============================================================================
# Using a standard alpha of 0.05, month, wind direction, and month:wind direction have a significant 
# effect on DewPoint Temperature. This means that both factors influence DewPoint Temperature, and the effect 
# of one factor is dependent on the other. Significant interactive effect is present.
# =============================================================================

## Pressure
# Run the ANOVA
model = sm.api.formula.ols('HourlySeaLevelPressure ~ Month + HourlyWindDirectionTier + HourlyWindDirectionTier:Month', data = dfs).fit()

# Extract our table
aov_table = sm.api.stats.anova_lm(model, typ=2)

# Print the table
print(aov_table)
# =============================================================================
# Using a standard alpha of 0.05, month, wind direction, and month:wind direction have a significant 
# effect on sea-level pressure. This means that both factors influence sea-level pressure, and the effect 
# of one factor is dependent on the other. Significant interactive effect is present.
# =============================================================================

#  visualize the probability distribution of January & July NE and SW samples in a single plot.
# Plotting the KDE Plot
# 2D density + marginal distribution:
sns.set_style("darkgrid")
sns.jointplot(data=dfs, x='HourlyDewPointTemperature', 
              y='HourlySeaLevelPressure', hue='HourlyWindDirectionTier', kind='kde',
              fill=True, joint_kws={'alpha': 0.4}, height=15)
plt.suptitle("Sample from Population data - Jan and July",y=1.08) # y= some height>1


sns.set_style("darkgrid")
sns.jointplot(data=dfs, x='HourlyDewPointTemperature', 
              y='HourlySeaLevelPressure', hue='Month', kind='kde',
              fill=True, joint_kws={'alpha': 0.4}, height=15)
plt.suptitle("Sample from Population data - Jan and July",y=1.08) # y= some height>1






# Is the Temperature wind correlated with Year
a=df[['HourlyDryBulbTemperature','Year']].dropna()
stats.pearsonr(a.HourlyDryBulbTemperature, a.Year)





# put legend outside of plot  in a sns plot - a bit complicated
sns.set_palette("bright")
ax=sns.histplot(binwidth=0.5, x="Month", hue="HourlyWindDirectionTier", 
             data=df, stat="density", multiple="fill")
ax.legend(handles=ax.legend_.legendHandles, labels=[t.get_text() for t in ax.legend_.texts],
          title=ax.legend_.get_title().get_text(),
          bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
#plt.tight_layout()
plt.show()


# put legend outside of plot  in a sns plot - a bit complicated
sns.set_palette("bright")
ax=sns.histplot(binwidth=1, x="Year", hue="HourlyWindDirectionTier", 
             data=df, stat="density", multiple="fill")
ax.legend(handles=ax.legend_.legendHandles, labels=[t.get_text() for t in ax.legend_.texts],
          title=ax.legend_.get_title().get_text(),
          bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
#plt.tight_layout()
plt.show()


#Remove Unknown wind direction
sns.set_palette("bright")
ax=sns.histplot(binwidth=1, x="Year", hue="HourlyWindDirectionTier", 
             data=df[df['HourlyWindDirectionTier']!='Unknown'], stat="density", multiple="fill")
ax.legend(handles=ax.legend_.legendHandles, labels=[t.get_text() for t in ax.legend_.texts],
          title=ax.legend_.get_title().get_text(),
          bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
#plt.tight_layout()
plt.show()



pd.crosstab(df.HourlyWindDirectionTier,df.Month)

pd.crosstab(df.HourlyWindDirectionTier,df.Year)

pd.crosstab(df.Year,df.HourlyWindDirectionTier).loc[:,['N','NE','SW','W','Unknown']].plot(figsize=(15,15))

pd.crosstab(df.HourlyWindDirectionTier,df.Year).loc['N',:].plot(kind='bar',title='Yearly Wind Counts',figsize=(15,15),alpha=0.5, legend=True)
pd.crosstab(df.HourlyWindDirectionTier,df.Year).loc['NE',:].plot(kind='bar',title='Yearly Wind Counts',figsize=(15,15),color='orange', alpha=0.5, legend=True)

pd.crosstab(df.HourlyWindDirectionTier,df.Year).loc['N',:].plot(kind='bar',title='Yearly Wind Counts',figsize=(15,15),alpha=0.5, legend=True)
pd.crosstab(df.HourlyWindDirectionTier,df.Year).loc['W',:].plot(kind='bar',title='Yearly Wind Counts',figsize=(15,15),color='green', alpha=0.5, legend=True)



pd.crosstab(df.HourlyWindDirectionTier,df.Year).loc['W',:].plot()
pd.crosstab(df.HourlyWindDirectionTier,df.Year).loc['SW',:].plot()
pd.crosstab(df.HourlyWindDirectionTier,df.Year).loc['Unknown',:].plot()



#plt.cm.  gives you cmap options

# =============================================================================
# cmap options supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 
# 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 
# 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 
# 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 
# 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 
# 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 
# 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 
# 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
#  'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 
#  'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 
#  'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 
#  'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 
#  'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 
#  'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 
#  'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 
#  'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 
#  'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 
#  'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 
#  'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 
#  'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 
#  'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 
#  'winter', 'winter_r'
# =============================================================================


p = df.plot(kind='scatter', 
                x='HourlyWetBulbTemperature', 
                y='HourlySeaLevelPressure', 
                c='HourlyDryBulbTemperature', cmap='rainbow',alpha=0.5,sharex=False, figsize=(15,10))
p.set_facecolor('black')


p = df.plot(kind='scatter', 
                x='HourlyDewPointTemperature', 
                y='HourlySeaLevelPressure', 
                c='HourlyDryBulbTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


sns.set_style("darkgrid")
fig_dims = (15, 10)
fig, ax = plt.subplots(figsize=fig_dims)
p=sns.scatterplot(x='HourlyDewPointTemperature', y='HourlySeaLevelPressure', data=df, hue='Month', ec=None)
p.set_facecolor('black')
#place legend outside top right corner of plot
plt.legend(bbox_to_anchor=(1.02, 1),loc='bottom left', borderaxespad=0)


# Plots of North winds
p = df[df['HourlyWindDirectionTier']=='N'].plot(kind='scatter', 
                x='Year', title ='North winds',
                y='HourlySeaLevelPressure', 
                c='HourlyDewPointTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


p = df[df['HourlyWindDirectionTier']=='N'].plot(kind='scatter', 
                x='Year', 
                y='HourlySeaLevelPressure', 
                c='HourlyDryBulbTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


p = df[df['HourlyWindDirectionTier']=='N'].plot(kind='scatter',
                x='Year', 
                y='Month', 
                c='HourlyDewPointTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


p = df[df['HourlyWindDirectionTier']=='N'].plot(kind='scatter',
                x='Year', 
                y='Month', 
                c='HourlySeaLevelPressure', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


# Plots of NorthEast winds
p = df[df['HourlyWindDirectionTier']=='NE'].plot(kind='scatter', 
                x='Year', title ='NorthEast winds',
                y='HourlySeaLevelPressure', 
                c='HourlyDewPointTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


p = df[df['HourlyWindDirectionTier']=='NE'].plot(kind='scatter', 
                x='Year', 
                y='HourlySeaLevelPressure', 
                c='HourlyDryBulbTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


p = df[df['HourlyWindDirectionTier']=='NE'].plot(kind='scatter',
                x='Year', 
                y='Month', 
                c='HourlyDewPointTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


p = df[df['HourlyWindDirectionTier']=='NE'].plot(kind='scatter',
                x='Year', 
                y='Month', 
                c='HourlySeaLevelPressure', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


p = df[df['HourlyWindDirectionTier']=='NE'].plot(kind='scatter',
                x='Year', 
                y='Month', 
                c='HourlyWindSpeed', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


# Plots of SouthWest winds
p = df[df['HourlyWindDirectionTier']=='SW'].plot(kind='scatter', 
                x='Year', title ='SouthWest winds',
                y='HourlySeaLevelPressure', 
                c='HourlyDewPointTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


p = df[df['HourlyWindDirectionTier']=='SW'].plot(kind='scatter', 
                x='Year', title ='SouthWest winds',
                y='HourlySeaLevelPressure', 
                c='HourlyDryBulbTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


# Plots of West winds
p = df[df['HourlyWindDirectionTier']=='W'].plot(kind='scatter', 
                x='Year', title ='West winds',
                y='HourlySeaLevelPressure', 
                c='HourlyDewPointTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')


p = df[df['HourlyWindDirectionTier']=='W'].plot(kind='scatter', 
                x='Year', title ='West winds',
                y='HourlySeaLevelPressure', 
                c='HourlyDryBulbTemperature', cmap='rainbow',alpha=0.5,sharex=False,figsize=(15,10))
p.set_facecolor('black')





# 2D density plot: *** Takes too long to plot
sns.kdeplot(data = df[df['HourlyWindSpeed']<100], x='HourlySeaLevelPressure', y='HourlyWindSpeed', cmap="Reds", shade=True)
plt.title('Windspeed vs Sea Level Pressure density graph', loc='left')
plt.show()

# 2D density plot: *** Takes too long to plot
sns.kdeplot(data = df, x='', y='HourlySeaLevelPressure', cmap="Reds", shade=True)
plt.title('HourlyDewPointTemperature vs Sea Level Pressure density graph', loc='left')
plt.show()


# 2D density + marginal distribution:
sns.set_style("darkgrid")
fig_dims = (15, 10)
fig, ax = plt.subplots(figsize=fig_dims)
p=sns.jointplot(x=df.HourlyDewPointTemperature, y=df.HourlySeaLevelPressure, cmap="Greens", shade=True, kind='kde')
p.set_facecolor('black')
plt.show()

# 2D density + marginal distribution:
w = df[df['HourlyWindSpeed']<100].HourlyWindSpeed
sns.set_style("darkgrid")
fig_dims = (15, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.jointplot(x=w, y=df.HourlySeaLevelPressure, cmap="Reds", shade=True, kind='kde')
plt.show()


p = df[df['HourlyWindSpeed']<100].plot(kind='scatter', 
                x='HourlyWetBulbTemperature', 
                y='HourlySeaLevelPressure', 
                c='HourlyWindSpeed', cmap='twilight_shifted',alpha=0.7,sharex=False,figsize=(15,10))
p.set_facecolor('black')




# =============================================================================
# # discrete color map cmap=plt.cm.get_cmap('gist_rainbow',8)
# df19742021.plot(kind='scatter', 
#                 x='HourlyDryBulbTemperature', 
#                 y='HourlySeaLevelPressure', 
#                 c='HourlyWindSpeed', 
#                 cmap=plt.cm.get_cmap('gist_rainbow',8),alpha=0.3,sharex=False)
# =============================================================================

# =============================================================================
# df19742021[df19742021['HourlyWindSpeed']<50].plot(kind='scatter', 
#                 x='HourlyWindDirection', 
#                 y='HourlyWindSpeed', 
#                 c='HourlyDryBulbTemperature', cmap='gist_rainbow_r',alpha=0.7,sharex=False,rot=90)
# =============================================================================

p = df19742021[df19742021['HourlyWindSpeed']<50].plot(kind='scatter', 
                x='HourlyWindDirection', 
                y='HourlyWindSpeed', 
                c='HourlyDewPointTemperature', 
                cmap='gist_rainbow_r',sharex=False,rot=90,figsize =(15,8))
p.set_facecolor('black')


p = df19742021.plot(kind='scatter', 
                x='HourlyWindDirection', 
                y='HourlyDewPointTemperature', 
                c='HourlyDryBulbTemperature', 
                cmap='gist_rainbow_r',sharex=False,rot=90,figsize =(15,8))
p.set_facecolor('black')


# =============================================================================
# # looking for santa ana conditions
# p = df19742021.plot(kind='scatter', 
#                 x='HourlyWindDirection', 
#                 ylim = (0,20),
#                 y='HourlyDewPointTemperature', 
#                 c='HourlyDryBulbTemperature', 
#                 cmap='gist_rainbow_r',sharex=False,rot=90,figsize =(15,8))
# p.set_facecolor('black')
# 
# =============================================================================


# =============================================================================
# sns.catplot(x="HourlyDewPointTemperature", y="HourlyWindDirection", data=df19742021)
# 
# sns.catplot(x="HourlyDewPointTemperature", y="HourlyWindDirection", jitter=False, data=df19742021)
# =============================================================================

sns.catplot(x="HourlyDewPointTemperature", y="HourlyWindDirection", kind="box", data=df19742021)



## Missing Data
df19742021.isnull().sum()
df19742021.isnull().mean()

# Smaller dataframe with key variables
df = df19742021.loc[:,['Year', 'Month', 'Week', 'Day',
                'HourlySeaLevelPressure',
               'HourlyDryBulbTemperature',
               'HourlyDewPointTemperature',
               'HourlyWetBulbTemperature',
               'HourlyPrecipitation',
               'HourlyWindSpeed','HourlyWindDirection',
               'HourlyVisibility']]

df.info()

df.isnull().sum()
df.isnull().mean()


df['HourlyWindDirection'].value_counts(normalize=True)
# replace string nan with NaN
df.loc[df.HourlyWindDirection=='nan', 'HourlyWindDirection']=np.NaN
# convert to float
df['HourlyWindDirection']= pd.to_numeric(df['HourlyWindDirection'],errors='coerce')

plt.figure(figsize=(14,6))
sns.heatmap(data=pd.DataFrame(df['HourlyWindDirection']).isnull(),cbar=False)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.xlabel('Missing HourlyWindDirection Data - Shown as Light Bars')
plt.show()





df['HourlySeaLevelPressure'].isnull().resample('A').sum()

max(df['HourlySeaLevelPressure'].isnull().resample('A').sum())

df['HourlySeaLevelPressure']['2001'].plot(kind='line')
df['HourlySeaLevelPressure']['1974'].plot(kind='line')

plt.figure(figsize=(14,6))
sns.heatmap(data=pd.DataFrame(df['HourlySeaLevelPressure']).isnull(),cbar=False)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.xlabel('Missing HourlySeaLevelPressure Data - Shown as Light Bars')
plt.show()



# df['HourlyWindDirection']=pd.to_numeric(df['HourlyWindDirection'],errors='coerce')
# df['HourlyWindDirection']=df['HourlyWindDirection'].astype(int,errors='ignore')



## hourly resample





sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlySeaLevelPressure", data=dfh)
sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlyDryBulbTemperature", data=dfh)
sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlyDryBulbTemperature", data=dfh)
sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlyWetBulbTemperature", data=dfh)
sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlyWindSpeed", data=dfh[dfh['HourlyWindSpeed']<50])
# sns.catplot(kind="box", x="HourlyWindDirectionTier", y="HourlyPrecipitation", data=dfh)


# put legend outside of plot  in a sns plot - a bit complicated
sns.set_palette("bright")
ax=sns.histplot(binwidth=0.5, x="Month", hue="HourlyWindDirectionTier", 
             data=dfh, stat="density", multiple="fill")
ax.legend(handles=ax.legend_.legendHandles, labels=[t.get_text() for t in ax.legend_.texts],
          title=ax.legend_.get_title().get_text(),
          bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.tight_layout()
plt.show()


# put legend outside of plot  in a sns plot - a bit complicated
ax=sns.histplot(binwidth=0.5, x="HourlyWindDirectionTier", hue="Month",
             data=dfh, stat="density", multiple="fill",palette='rainbow')
ax.legend(handles=ax.legend_.legendHandles, labels=[t.get_text() for t in ax.legend_.texts],
          title=ax.legend_.get_title().get_text(),
          bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.tight_layout()
plt.show()


dfh['HourlyWindDirectionTier'].value_counts(normalize=True)
