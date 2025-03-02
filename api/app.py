from flask import Flask, jsonify
import logging
import os
import sys
import time
import traceback

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendations import recommendation_bp
from utils.config_manager import ConfigManager

# 初始化Flask应用
app = Flask(__name__)

# 加载配置
config = ConfigManager().get_config()

# 注册蓝图
app.register_blueprint(recommendation_bp, url_prefix='/api')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(
            config.get('logging.file.path', 'logs'),
            f"api_{time.strftime('%Y%m%d')}.log"
        ))
    ]
)

logger = logging.getLogger(__name__)

# 全局错误处理
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"未捕获的异常: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({"error": "服务器内部错误", "details": str(e)}), 500

# 健康检查接口
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": config.get('api.version', '1.0.0')
    })

# 404处理
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "请求的资源不存在"}), 404

if __name__ == '__main__':
    # 获取配置中的API设置
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 5000)
    debug = config.get('api.debug', False)
    
    logger.info(f"API服务启动于 http://{host}:{port}")
    app.run(host=host, port=port, debug=debug) 