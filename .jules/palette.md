# Palette's Journal - Critical Learnings Only

## 2026-11-03 - Financial Chart Spatial Controls & Keyboard Accessibility
**Learning:** Financial candlestick canvas visualizations often suffer from navigation trapping where users pan/zoom out into empty space without an obvious way back. Standard desktop charting patterns (TradingView, MT5) rely on double-clicking the canvas to reset spatial view (`1.0x` zoom, `0` pan offset) and keyboard shortcuts (`+`, `-`, `0`).
**Action:** Always bind both canvas `dblclick` and window `keydown` (`+`, `-`, `0`) to chart zoom/reset handlers alongside `:focus-visible` styling, allowing keyboard-only and mouse-only power traders to navigate without UI friction.
