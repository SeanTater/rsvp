import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('logs.dat', delimiter = " ") 
df.columns=["dimensions", "ratio", "mse", "psnr"]

mx = max(df.dimensions)
df['colors'] = df.apply(lambda row: str(row.dimensions/mx), axis=1)


fig, ax = plt.subplots()
sc = plt.scatter(df.ratio, df.psnr, s=500, c=df.colors)

ax.set(xlabel='compression ratio (%)', ylabel='PSNR',
       title='Compression Performance vs PSNR')
ax.grid()
fig.colorbar(sc, ax=ax)
plt.set_cmap('gray')
plt.clim(0,mx)
fig.savefig("performance.png")
