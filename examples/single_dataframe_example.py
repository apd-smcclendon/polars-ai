import polars as pl
from polarsAI import PolarsAI

df = pl.DataFrame(
	{"A": [1, 2, 3, 4, 5],
	"fruits": ["banana", "banana", "apple", "apple", "banana"],
	"B": [5, 4, 3, 2, 1], 
	"cars": ["beetle", "audi", "beetle", "beetle", "beetle"],}
	)

obj=PolarsAI(llm_type='OpenAI')   

obj.run(data_frame = df, prompt = 'what is the sum of the A column?')  