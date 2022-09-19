import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

 

model = pickle.load(open('FineTech_app_ML_model.pickle','rb'))
st.title('Directing Customers to Subsciption Through Financial AppBehavior Analysis')


dayofweek = st.number_input('dayofweek')
hour = st.number_input('hour')
age = st.number_input('age')
numscreens = st.number_input('numscreens')
minigame = st.number_input('minigame')
used_premium_feature = st.number_input('used_premium_feature')
liked = st.number_input('liked')
location = st.number_input('location')
Institutions = st.number_input('Institutions')
VerifyPhone = st.number_input('VerifyPhone')
BankVerification = st.number_input('BankVerification')
VerifyDateOfBirth = st.number_input('VerifyDateOfBirth')
ProfilePage = st.number_input('ProfilePage')
VerifyCountry = st.number_input('VerifyCountry')
Cycle = st.number_input('Cycle')
idscreen = st.number_input('idscreen')
Splash = st.number_input('Splash')
RewardsContainer = st.number_input('RewardsContainer')
EditProfile = st.number_input('EditProfile')
Finances = st.number_input('Finances')
Alerts = st.number_input('Alerts')
Leaderboard = st.number_input('Leaderboard')
VerifyMobile = st.number_input('VerifyMobile')
VerifyHousing = st.number_input('VerifyHousing')
RewardDetail = st.number_input('RewardDetail')
VerifyHousingAmount = st.number_input('VerifyHousingAmount')
ProfileMaritalStatus = st.number_input('ProfileMaritalStatus')
ProfileChildren = st.number_input('ProfileChildren')
ProfileEducation = st.number_input('ProfileEducation')
ProfileEducationMajor = st.number_input('ProfileEducationMajor')
Rewards = st.number_input('Rewards')
AccountView = st.number_input('AccountView')
VerifyAnnualIncome = st.number_input('VerifyAnnualIncome')
VerifyIncomeType = st.number_input('VerifyIncomeType')
ProfileJobTitle = st.number_input('ProfileJobTitle')
Login = st.number_input('Login')
ProfileEmploymentLength = st.number_input('ProfileEmploymentLength')
WebView = st.number_input('WebView')
SecurityModal = st.number_input('SecurityModal')
ResendToken = st.number_input('ResendToken')
TransactionList = st.number_input('TransactionList')
NetworkFailure = st.number_input('NetworkFailure')
ListPicker = st.number_input('ListPicker')
remain_screen_list = st.number_input('remain_screen_list')
saving_screens_count = st.number_input('saving_screens_count')
credit_screens_count = st.number_input('credit_screens_count')
cc_screens_count = st.number_input('cc_screens_count')
loan_screens_count = st.number_input('loan_screens_count')

if st.button('Predict'):
    preds=model.predict(np.array([[ dayofweek, hour, age, numscreens, minigame,
       used_premium_feature, liked, location, Institutions,
       VerifyPhone, BankVerification, VerifyDateOfBirth, ProfilePage,
       VerifyCountry, Cycle, idscreen, Splash, RewardsContainer,
       EditProfile, Finances, Alerts, Leaderboard, VerifyMobile,
       VerifyHousing, RewardDetail, VerifyHousingAmount,
       ProfileMaritalStatus, ProfileChildren , ProfileEducation,
       ProfileEducationMajor, Rewards, AccountView, VerifyAnnualIncome,
       VerifyIncomeType, ProfileJobTitle, Login,
       ProfileEmploymentLength, WebView, SecurityModal, ResendToken,
       TransactionList, NetworkFailure, ListPicker, remain_screen_list,
       saving_screens_count, credit_screens_count, cc_screens_count,
       loan_screens_count]]))
    
    if(preds==[1]):
        st.success('Congratulations!')
        st.write('This Customer Will Buy the Subscription.')
    else:
        st.write('This Customer Will Not Buy the Subscription.')
    