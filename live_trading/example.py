"""
实时交易执行引擎使用示例
演示如何使用LiveTradingEngine进行实时交易
"""

import asyncio
import json
from datetime import datetime
from live_trading.engine import LiveTradingEngine, TradingSignal, EmergencyLevel
from live_trading.order_management import Order, OrderType, OrderSide, TimeInForce


async def main():
    """主函数示例"""
    
    # 配置参数
    config = {
        'websocket': {
            'exchange': 'binance',
            'testnet': True,
            'auto_reconnect': True,
            'heartbeat_interval': 30
        },
        'orders': {
            'exchange': 'binance',
            'testnet': True,
            'api_key': 'your_api_key',
            'api_secret': 'your_api_secret',
            'max_orders_per_symbol': 10,
            'max_order_amount': 100000,
            'order_timeout': 300
        },
        'positions': {
            'max_position_size': 1000000,
            'max_leverage': 10,
            'risk_free_rate': 0.02
        },
        'sync': {
            'sync_interval': 60,
            'max_sync_lag': 300
        },
        'trailing_stops': {
            'default_distance': 2.0,  # 2%
            'min_distance': 0.5,      # 0.5%
            'max_distance': 10.0      # 10%
        },
        'logging': {
            'log_dir': 'logs',
            'log_level': 'INFO',
            'max_file_size': 10 * 1024 * 1024,
            'backup_count': 5
        }
    }
    
    # 创建交易引擎
    engine = LiveTradingEngine(config)
    
    # 注册事件处理器
    await register_event_handlers(engine)
    
    try:
        # 启动引擎
        print("启动实时交易引擎...")
        
        # 在后台启动引擎
        engine_task = asyncio.create_task(engine.start())
        
        # 等待引擎启动
        await asyncio.sleep(5)
        
        # 模拟交易操作
        await simulate_trading(engine)
        
        # 运行一段时间
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("收到中断信号，正在停止...")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 停止引擎
        print("停止交易引擎...")
        await engine.stop()


async def register_event_handlers(engine: LiveTradingEngine):
    """注册事件处理器"""
    
    async def on_order_filled(data):
        """订单成交处理"""
        order = data.get('order')
        print(f"✅ 订单成交: {order.symbol} {order.side.value} {order.filled_quantity}@{order.average_price}")
    
    async def on_position_updated(data):
        """仓位更新处理"""
        position = data.get('position')
        if position and not position.is_flat:
            print(f"📊 仓位更新: {position.symbol} {position.side.value} {position.size} "
                  f"入场价:{position.entry_price} 当前价:{position.market_price} "
                  f"盈亏:{position.unrealized_pnl:.2f}")
    
    async def on_stop_loss_triggered(data):
        """止损触发处理"""
        position = data.get('position')
        print(f"🛑 止损触发: {position.symbol} @ {data.get('current_price')}")
    
    async def on_risk_warning(data):
        """风险警告处理"""
        risk_type = data.get('risk_type')
        symbol = data.get('symbol')
        print(f"⚠️  风险警告: {symbol} - {risk_type}")
    
    # 注册处理器
    engine.register_handler('order_filled', on_order_filled)
    engine.register_handler('position_updated', on_position_updated)
    engine.register_handler('stop_loss_triggered', on_stop_loss_triggered)
    engine.register_handler('position_risk_warning', on_risk_warning)


async def simulate_trading(engine: LiveTradingEngine):
    """模拟交易操作"""
    
    print("🚀 开始模拟交易...")
    
    # 等待系统稳定
    await asyncio.sleep(2)
    
    # 示例1: 市价买入订单
    signal1 = TradingSignal(
        symbol='BTCUSDT',
        action='BUY',
        price=50000.0,  # 市价单会忽略此价格
        quantity=0.001,
        confidence=0.8,
        timestamp=datetime.now(),
        strategy_id='demo_strategy',
        metadata={'signal_type': 'breakout', 'timeframe': '1h'}
    )
    
    await engine.submit_signal(signal1)
    print(f"📤 提交买入信号: {signal1.symbol} {signal1.quantity}")
    
    # 等待订单处理
    await asyncio.sleep(3)
    
    # 示例2: 限价卖出订单
    signal2 = TradingSignal(
        symbol='BTCUSDT',
        action='SELL',
        price=51000.0,  # 限价
        quantity=0.0005,
        confidence=0.7,
        timestamp=datetime.now(),
        strategy_id='demo_strategy',
        metadata={'signal_type': 'take_profit', 'target_pct': 2.0}
    )
    
    await engine.submit_signal(signal2)
    print(f"📤 提交限价卖出信号: {signal2.symbol} {signal2.quantity}@{signal2.price}")
    
    # 等待处理
    await asyncio.sleep(2)
    
    # 示例3: 查看引擎状态
    print(f"🔍 引擎状态: {engine.get_state().value}")
    print(f"📈 性能指标: {json.dumps(engine.get_performance_metrics(), indent=2)}")
    
    # 示例4: 查看仓位信息
    positions = await engine.position_manager.get_active_positions()
    print(f"💼 活跃仓位数量: {len(positions)}")
    
    for position in positions:
        print(f"   {position.symbol}: {position.side.value} {position.size} "
              f"盈亏: {position.unrealized_pnl:.2f} ({position.pnl_percentage:.2f}%)")
    
    # 示例5: 查看订单状态
    active_orders = await engine.order_manager.get_active_orders()
    print(f"📋 活跃订单数量: {len(active_orders)}")
    
    for order in active_orders:
        print(f"   {order.symbol}: {order.order_type.value} {order.side.value} "
              f"{order.quantity} 状态: {order.status.value}")
    
    # 示例6: 设置追踪止损
    if positions:
        position = positions[0]
        stop_id = engine.trailing_stop_manager.add_trailing_stop(
            symbol=position.symbol,
            side='long' if position.is_long else 'short',
            entry_price=position.entry_price,
            trailing_distance=2.0,  # 2%
            is_percentage=True
        )
        print(f"🔒 设置追踪止损: {stop_id}")
    
    # 示例7: 模拟紧急情况测试
    await asyncio.sleep(5)
    print("⚠️  模拟风险警告...")
    await engine.emergency_stop(EmergencyLevel.MEDIUM, "模拟测试紧急停止")
    
    # 等待处理
    await asyncio.sleep(2)
    
    # 恢复交易
    print("✅ 恢复交易...")
    await engine.resume_trading()


async def demo_manual_order():
    """演示手动创建订单"""
    
    print("\n=== 手动订单创建示例 ===")
    
    # 创建限价买入订单
    buy_order = Order(
        symbol='ETHUSDT',
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=0.1,
        price=3000.0,
        time_in_force=TimeInForce.GTC,
        strategy_id='manual_strategy',
        metadata={'note': '手动创建的订单'}
    )
    
    print(f"创建订单: {buy_order.to_dict()}")
    
    # 创建止损订单
    stop_order = Order(
        symbol='ETHUSDT',
        order_type=OrderType.STOP,
        side=OrderSide.SELL,
        quantity=0.1,
        stop_price=2900.0,
        time_in_force=TimeInForce.GTC,
        strategy_id='manual_strategy',
        metadata={'note': '止损订单'}
    )
    
    print(f"创建止损订单: {stop_order.to_dict()}")


async def demo_position_management():
    """演示仓位管理"""
    
    print("\n=== 仓位管理示例 ===")
    
    # 模拟成交数据
    fill_data = {
        'symbol': 'BTCUSDT',
        'side': 'buy',
        'quantity': 0.001,
        'price': 50000.0,
        'fee': 0.05,
        'fee_currency': 'USDT',
        'order_id': 'test_order_1',
        'trade_id': 'test_trade_1',
        'strategy_id': 'demo_strategy'
    }
    
    print(f"模拟成交: {fill_data}")
    
    # 显示投资组合摘要
    portfolio_summary = {
        'total_positions': 1,
        'total_realized_pnl': 100.0,
        'total_unrealized_pnl': 50.0,
        'total_pnl': 150.0,
        'total_notional_value': 50000.0,
        'win_rate': 65.0,
        'sharpe_ratio': 1.2
    }
    
    print(f"投资组合摘要: {json.dumps(portfolio_summary, indent=2)}")


if __name__ == "__main__":
    print("=== 实时交易执行引擎演示 ===")
    print("注意: 这是一个演示程序，请确保在测试环境中运行")
    print("=" * 50)
    
    # 运行主程序
    asyncio.run(main())
    
    # 运行其他演示
    asyncio.run(demo_manual_order())
    asyncio.run(demo_position_management())