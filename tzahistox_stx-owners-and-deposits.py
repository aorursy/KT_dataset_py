from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

NumberOfTopUsersToCheck = 500

query_to = """
SELECT 
  to_address AS address,
  SUM(CAST(value as NUMERIC)) AS to_val_sum
FROM
  `bigquery-public-data.ethereum_blockchain.token_transfers` AS to_transfers 
WHERE TRUE
  AND to_transfers.token_address = '0x006bea43baa3f7a6f765f14f10a1a1b08334ef45'
  AND CAST(to_transfers.value AS NUMERIC) > 0
GROUP BY
  address
ORDER BY
  to_val_sum DESC
"""

query_from = """
SELECT 
  from_address AS address,
  SUM(CAST(value as NUMERIC)) AS from_val_sum
FROM
  `bigquery-public-data.ethereum_blockchain.token_transfers` AS from_transfers 
WHERE TRUE
  AND from_transfers.token_address = '0x006bea43baa3f7a6f765f14f10a1a1b08334ef45'
  AND CAST(from_transfers.value AS NUMERIC) > 0
GROUP BY
  address
ORDER BY
  from_val_sum DESC
"""
query_to_job = client.query(query_to)

iterator_to = query_to_job.result(timeout=30)
rows_to = list(iterator_to)

# Transform the rows into a nice pandas dataframe
df_to = pd.DataFrame(data=[list(x.values()) for x in rows_to], columns=list(rows_to[0].keys()))
df_to.name = 'df_to'

# Look at the first 50
#df_to.head(50)

query_from_job = client.query(query_from)

iterator_from = query_from_job.result(timeout=30)
rows_from = list(iterator_from)

# Transform the rows into a nice pandas dataframe
df_from = pd.DataFrame(data=[list(x.values()) for x in rows_from], columns=list(rows_from[0].keys()))
df_from.name = 'df_from'

# Look at the first 50
#df_from.head(50)

df_to_slice = df_to.head(NumberOfTopUsersToCheck)
df_from_slice = df_from.head(NumberOfTopUsersToCheck)

df_balances = pd.merge(df_to_slice,df_from_slice,on=['address'])

df_balances['balance'] = df_balances['to_val_sum'] - df_balances['from_val_sum']
df_balances_sorted = df_balances.sort_values(by='balance', ascending=False)
df_balances_sorted[df_balances_sorted['balance']>0]

top_addresses = list(df_balances_sorted['address'])

query_contracts = """
SELECT 
  address
FROM
  `bigquery-public-data.ethereum_blockchain.contracts` AS contracts
WHERE
  contracts.bytecode = '0x6060604052600436106100485763ffffffff60e060020a6000350416631d9082c4811461004d578063521eb27314610084578063e1aed8a01461010a578063fe35530c1461012f575b600080fd5b341561005857600080fd5b610082600160a060020a03600480358216916020918201803592908101803590921691013561014e565b005b341561008f57600080fd5b6100976101e3565b6040518085600160a060020a0316600160a060020a0316815260200184600160a060020a0316600160a060020a0316815260200183600160a060020a0316600160a060020a0316815260200182600160a060020a0316600160a060020a0316815260200194505050505060405180910390f35b341561011557600080fd5b61008260048035600160a060020a03169060200135610223565b341561013a57600080fd5b610082600160a060020a03600435166102a4565b73c8b55c7ad00fb9b933b0a016c6cebceea0293bb9631557a52f60008686868660405160e060020a63ffffffff8816028152600401948552600160a060020a0393841684166020958601908152850192835290831690921690830190815282019081520160006040518083038186803b15156101c957600080fd5b6102c65a03f415156101da57600080fd5b50505050505050565b600080800154600182015460028301546003840154600160a060020a036101009590950a9384900485169492849004831693918290048316929190041684565b73c8b55c7ad00fb9b933b0a016c6cebceea0293bb963daf011426000848460405160e060020a63ffffffff8616028152600401928352600160a060020a03918216909116602092830190815282019081520160006040518083038186803b151561028c57600080fd5b6102c65a03f4151561029d57600080fd5b5050505050565b73c8b55c7ad00fb9b933b0a016c6cebceea0293bb9632eb9e5d760008360405160e060020a63ffffffff8516028152600401918252600160a060020a039081161660209182019081520160006040518083038186803b151561030557600080fd5b6102c65a03f4151561031657600080fd5b505050505600a165627a7a7230582069ca1cca38613ca9f68e12db866939f07bf9490f54ca67c98a85c1233c5305130029'
"""

query_contracts_job = client.query(query_contracts)

iterator_contracts = query_contracts_job.result(timeout=30)
rows_contracts = list(iterator_contracts)

# Transform the rows into a nice pandas dataframe
df_contracts = pd.DataFrame(data=[list(x.values()) for x in rows_contracts], columns=list(rows_contracts[0].keys()))

stox_wallet_addresses = list(df_contracts['address'])

query_logs = """
SELECT 
  transaction_hash,
  address,
  topics,
  data
FROM
  `bigquery-public-data.ethereum_blockchain.logs` AS logs
WHERE TRUE
  AND logs.address = '0x006bea43baa3f7a6f765f14f10a1a1b08334ef45'
  AND logs.topics[OFFSET(0)] = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
  AND logs.data != '0x0000000000000000000000000000000000000000000000000000000000000000'
"""
pd.set_option('display.max_colwidth', 100)
query_logs_job = client.query(query_logs)

iterator_logs = query_logs_job.result(timeout=30)
rows_logs = list(iterator_logs)

# Transform the rows into a nice pandas dataframe
df_logs = pd.DataFrame(data=[list(x.values()) for x in rows_logs], columns=list(rows_logs[0].keys()))

topics = list(df_logs['topics'])
hashes = list(df_logs['transaction_hash'])

top_users = []
full_data = []

for ind in range(len(topics)):
    formatted_topic_from = str(hex(int(topics[ind][1],16)))
    formatted_topic_to = str(hex(int(topics[ind][2],16)))
    if ((formatted_topic_from in top_addresses) and (formatted_topic_to in stox_wallet_addresses) and (formatted_topic_from not in top_users)):
        top_users.append(formatted_topic_from)
        full_data.append([hashes[ind],formatted_topic_from])
        
print("# of top STX holders: " + str(len(top_addresses)))        
print("# of the top holders who transfered funds to a Stox wallet: " + str(len(full_data)))        
print('Addresses: \n' + str(full_data))  
