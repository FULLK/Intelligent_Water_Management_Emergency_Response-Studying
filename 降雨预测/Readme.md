在 Conda 环境下，生成 `requirements.txt` 文件的方式与使用 pip 略有不同。Conda 主要用于环境管理和包管理，因此它有自己的方法来记录和重现环境。不过，如果你确实需要一个类似于 `requirements.txt` 的文件，你可以通过以下几种方式来实现。

### 方法一：导出 Conda 环境到 YAML 文件

Conda 推荐使用 `.yml` 文件来保存环境配置，包括所有依赖项及其版本。这是最直接的方法，可以确保完全再现你的环境。

1. **激活你的 Conda 环境**（如果尚未激活）：
   ```bash
   conda activate your_env_name
   ```

2. **导出当前环境到 YAML 文件**：
   ```bash
   conda env export > environment.yml
   ```

3. **创建新环境时使用此 YAML 文件**：
   ```bash
   conda env create -f environment.yml
   ```

这种方法会将所有的依赖关系（包括 Python 版本、Conda 包和 pip 包）都记录下来。

### 方法二：结合 Conda 和 pip 生成 `requirements.txt`

如果你的环境中既有 Conda 安装的包也有 pip 安装的包，并且你确实需要一个 `requirements.txt` 文件，你可以采取以下步骤：

1. **激活你的 Conda 环境**：
   ```bash
   conda activate your_env_name
   ```

2. **导出仅包含 pip 包的 `requirements.txt`**：
   ```bash
   pip freeze > requirements.txt
   ```

3. **如果需要同时列出 Conda 和 pip 包**，你可以先导出 Conda 包列表，然后手动或脚本处理将其转换为 `requirements.txt` 格式，或者只关注 pip 包部分。

4. **为了更精确地控制哪些包应该出现在 `requirements.txt` 中**，你可以考虑只列出那些通过 pip 安装的包，而让 Conda 负责管理其他依赖。

