# SEBRA Payments Q1 of 2025

This application allows for interacting with the public database for all budget spendings issued by the Bulgarian government for the period of 01.01.2025 to 31.03.2025 of above 5,000 BGN via SEBRA (system for electronic budget payments). The data is freely available [here](https://data.egov.bg/data/view/57f1e2e7-b235-45e8-94c4-4d69f0b1a690) and it contains information for a bit over 100,000 transactions. The features I kept are the following:

- Date of the transaction
- Amount in BGN
- Name of the client
- Name of the organization
- Name of the primary organization
- SEBRA pay code, which categorizes the payment in 12 categories

The other columns are omitted due to incosistencies in the data that require a lot of manual cleaning. I arranged the data into a few tables such that it satisfies 3NF. The entity relationship diagram is the following:

![](assets/sebra/ER.svg)
