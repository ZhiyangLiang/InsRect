import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
data = pd.DataFrame.from_dict(
    {
        "step": [random.randint(1, 100) for _ in range(5000)],
        "reward": [random.random() for _ in range(5000)],
    }
)

sns.lineplot(data=data, x="step", y="reward", color="orange")
plt.show()
