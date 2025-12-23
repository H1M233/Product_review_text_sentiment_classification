## 中文数据集 (Chinese)
训练集大小：10000 | 测试集大小：2500

<table>
  <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">准确率</th>
    <th colspan="3">负面</th>
    <th colspan="3">正面</th>
  </tr>
  <tr>
    <th>精准率</th>
    <th>召回率</th>
    <th>F1</th>
    <th>精准率</th>
    <th>召回率</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>Naive Bayes</td>
    <td>0.76</td>
    <td>0.76</td>
    <td>0.77</td>
    <td>0.76</td>
    <td>0.76</td>
    <td>0.76</td>
    <td>0.76</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>0.76</td>
    <td>0.76</td>
    <td>0.75</td>
    <td>0.76</td>
    <td>0.75</td>
    <td>0.76</td>
    <td>0.76</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.73</td>
    <td>0.73</td>
    <td>0.75</td>
    <td>0.74</td>
    <td>0.74</td>
    <td>0.72</td>
    <td>0.73</td>
  </tr>
  <tr>
    <td>CNN</td>
    <td>0.74</td>
    <td>0.76</td>
    <td>0.69</td>
    <td>0.72</td>
    <td>0.71</td>
    <td>0.79</td>
    <td>0.75</td>
  </tr>
  <tr>
    <td>BERT</td>
    <td>0.77</td>
    <td>0.84</td>
    <td>0.67</td>
    <td>0.75</td>
    <td>0.73</td>
    <td>0.87</td>
    <td>0.79</td>
  </tr>
</table>

### 使用模型预测

预测内容：绝对精典的动画片，它是我们八十年代人童年的烙印，直到现在它依然是我的最爱！不仅是它那丰富的故事内容，而且留给我的那一份难忘的童年情怀！这就是优秀动画片对人的真正魅力所在，它们在让人欣赏娱乐的同时也为留住了对岁月的印象。

预测结果：
| Naive Bayes | SVM | Random Forest | CNN | BERT |
|:---:|:---:|:---:|:---:|:---:|
| 正面 | 正面 | 正面 | 正面 | 正面 |

---

## 英文数据集 (English)
训练集大小：9985 | 测试集大小：2500

<table>
  <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">准确率</th>
    <th colspan="3">负面</th>
    <th colspan="3">正面</th>
  </tr>
  <tr>
    <th>精准率</th>
    <th>召回率</th>
    <th>F1</th>
    <th>精准率</th>
    <th>召回率</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>Naive Bayes</td>
    <td>0.85</td>
    <td>0.86</td>
    <td>0.82</td>
    <td>0.84</td>
    <td>0.83</td>
    <td>0.87</td>
    <td>0.85</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>0.87</td>
    <td>0.87</td>
    <td>0.87</td>
    <td>0.87</td>
    <td>0.87</td>
    <td>0.87</td>
    <td>0.87</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.82</td>
    <td>0.79</td>
    <td>0.86</td>
    <td>0.83</td>
    <td>0.85</td>
    <td>0.78</td>
    <td>0.81</td>
  </tr>
  <tr>
    <td>CNN</td>
    <td>0.86</td>
    <td>0.84</td>
    <td>0.88</td>
    <td>0.86</td>
    <td>0.87</td>
    <td>0.84</td>
    <td>0.85</td>
  </tr>
  <tr>
    <td>BERT</td>
    <td>0.91</td>
    <td>0.89</td>
    <td>0.93</td>
    <td>0.91</td>
    <td>0.93</td>
    <td>0.89</td>
    <td>0.91</td>
  </tr>
</table>

### 使用模型预测

预测内容：I originally had a Brookstone pedometer that was not accurate at all (like for every 1 step, it counted 2-3), so someone told me about this one and I ordered it right away. It is soooooooooooooooo accurate. For the whole day, it might be off by like a few steps, but nothing to worry about. The only thing I dont like about it is that it is huge compared to other pedometers, but I would rather have it big and accurate than not. I would definitely recommend this pedometer to everyone.

预测结果：
| Naive Bayes | SVM | Random Forest | CNN | BERT |
|:---:|:---:|:---:|:---:|:---:|
| 正面 | 正面 | 负面 | 正面 | 正面 |
  
---

## 双语数据集 (Bilingual)
训练集大小：19985 | 测试集大小：5000

<table>
  <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">准确率</th>
    <th colspan="3">负面</th>
    <th colspan="3">正面</th>
  </tr>
  <tr>
    <th>精准率</th>
    <th>召回率</th>
    <th>F1</th>
    <th>精准率</th>
    <th>召回率</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>Naive Bayes</td>
    <td>0.78</td>
    <td>0.79</td>
    <td>0.77</td>
    <td>0.78</td>
    <td>0.77</td>
    <td>0.79</td>
    <td>0.78</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>0.79</td>
    <td>0.79</td>
    <td>0.80</td>
    <td>0.79</td>
    <td>0.79</td>
    <td>0.79</td>
    <td>0.79</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.77</td>
    <td>0.77</td>
    <td>0.77</td>
    <td>0.77</td>
    <td>0.77</td>
    <td>0.77</td>
    <td>0.77</td>
  </tr>
  <tr>
    <td>CNN</td>
    <td>0.74</td>
    <td>0.68</td>
    <td>0.91</td>
    <td>0.78</td>
    <td>0.86</td>
    <td>0.57</td>
    <td>0.68</td>
  </tr>
  <tr>
    <td>BERT</td>
    <td>0.81</td>
    <td>0.83</td>
    <td>0.79</td>
    <td>0.81</td>
    <td>0.80</td>
    <td>0.83</td>
    <td>0.81</td>
  </tr>
</table>

### 使用模型预测：

#### 中文
预测内容：绝对精典的动画片，它是我们八十年代人童年的烙印，直到现在它依然是我的最爱！不仅是它那丰富的故事内容，而且留给我的那一份难忘的童年情怀！这就是优秀动画片对人的真正魅力所在，它们在让人欣赏娱乐的同时也为留住了对岁月的印象。

预测结果：
| Naive Bayes | SVM | Random Forest | CNN | BERT | 综合 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 负面 | 负面 | 正面 | 负面 | 正面 | 负面 |

#### 英文
预测内容：I originally had a Brookstone pedometer that was not accurate at all (like for every 1 step, it counted 2-3), so someone told me about this one and I ordered it right away. It is soooooooooooooooo accurate. For the whole day, it might be off by like a few steps, but nothing to worry about. The only thing I dont like about it is that it is huge compared to other pedometers, but I would rather have it big and accurate than not. I would definitely recommend this pedometer to everyone.

预测结果：
| Naive Bayes | SVM | Random Forest | CNN | BERT | 综合
|:---:|:---:|:---:|:---:|:---:|:---:|
| 正面 | 正面 | 负面 | 负面 | 正面 | 正面 |
