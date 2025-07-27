import pandas as pd

# প্রথম CSV ফাইল থেকে v1 ও v2 কলাম পড়া
df1 = pd.read_csv('spam_final.csv', usecols=['v1', 'v2'])

# দ্বিতীয় CSV ফাইল থেকে v1 ও v2 কলাম পড়া
df2 = pd.read_csv('spam_final_final.csv', usecols=['v1', 'v2'])

# দুইটা একসাথে মিক্স করা
combined = pd.concat([df1, df2], ignore_index=True)

# নতুন ফাইলে সেভ করা
combined.to_csv('data.csv', index=False)
