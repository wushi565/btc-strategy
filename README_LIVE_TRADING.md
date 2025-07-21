# 比特币实时交易系统

这是一个基于技术指标的比特币实时交易系统，支持在币安交易所进行自动化交易。

## 功能特性

- **实时数据获取**: 从币安交易所获取实时K线数据
- **技术指标计算**: 支持Supertrend、DEMA、ADX等技术指标
- **信号生成**: 基于技术指标生成买卖信号
- **风险管理**: 支持止损和目标价格设置
- **仓位管理**: 自动计算仓位大小和风险控制
- **实时监控**: 提供实时交易状态显示

## 系统要求

- Python 3.7+
- 币安API密钥
- 稳定的网络连接

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

### 1. 配置文件 (config.yaml)

```yaml
trading:
  symbol: "BTC/USDT"      # 交易对
  timeframe: "1h"         # 时间周期
  is_futures: true        # 是否期货交易

network:
  enable_proxy: false     # 是否启用代理
  http_proxy: "socks5://127.0.0.1:7890"  # 代理地址

indicators:
  dema144_len: 144        # DEMA144周期
  dema169_len: 169        # DEMA169周期
  atr_period: 34          # ATR周期
  atr_multiplier: 3.0     # ATR乘数
  adx_period: 14          # ADX周期
  adx_threshold: 20       # ADX阈值

signals:
  risk_reward_ratio: 3.0  # 风险回报比
  strategy: "Supertrend和DEMA策略"  # 策略名称

backtest:
  initial_capital: 10000.0  # 初始资金
  leverage: 1.0             # 杠杆
  risk_per_trade: 0.02      # 每笔交易风险比例
```

### 2. API配置 (api_config.yaml)

```yaml
# 币安API配置
api_key: "your_api_key_here"
api_secret: "your_api_secret_here"
testnet: false  # 是否使用测试网络
sandbox: false  # 是否使用沙盒环境
```

## 使用步骤

### 1. 获取币安API密钥

1. 登录币安账户
2. 进入API管理页面
3. 创建新的API密钥
4. 确保API密钥具有交易权限
5. 建议设置IP白名单

### 2. 配置API密钥

编辑 `api_config.yaml` 文件，填入您的API密钥：

```yaml
api_key: "您的API密钥"
api_secret: "您的API密钥"
```

### 3. 启动实时交易

```bash
python live_trading.py
```

## 交易策略

### Supertrend和DEMA策略

**开多条件:**
1. Supertrend出现买入信号
2. 价格突破DEMA144和DEMA169
3. 止损设置在DEMA169线上

**开空条件:**
1. Supertrend出现卖出信号
2. 价格跌破DEMA144和DEMA169
3. 止损设置在DEMA144线上

### Supertrend、DEMA和ADX策略

在基础策略基础上增加ADX过滤条件：
- ADX > 20 表示趋势强度足够

## 风险控制

- **止损**: 自动设置止损价格
- **目标价格**: 基于风险回报比设置目标价格
- **仓位大小**: 根据账户余额和风险比例自动计算
- **最大风险**: 每笔交易最大风险为账户余额的2%

## 监控和日志

- **实时状态**: 每分钟显示交易状态
- **日志文件**: 所有操作记录在 `live_trading.log`
- **交易统计**: 显示总交易次数、胜率等信息

## 安全注意事项

1. **API密钥安全**: 不要泄露API密钥
2. **IP白名单**: 建议设置API密钥的IP白名单
3. **测试环境**: 首次使用建议在测试网络测试
4. **资金管理**: 不要投入过多资金，建议从小额开始
5. **监控运行**: 定期检查系统运行状态

## 故障排除

### 常见问题

1. **连接失败**: 检查网络连接和代理设置
2. **API错误**: 检查API密钥是否正确
3. **权限不足**: 确保API密钥有交易权限
4. **资金不足**: 检查账户余额

### 日志查看

```bash
tail -f live_trading.log
```

## 免责声明

- 本系统仅供学习和研究使用
- 加密货币交易存在高风险
- 请根据自身风险承受能力谨慎使用
- 作者不对任何交易损失负责

## 技术支持

如有问题，请查看日志文件或联系技术支持。 