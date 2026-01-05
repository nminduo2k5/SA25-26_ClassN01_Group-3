"""
ARCHITECTURE MANAGER
Quản lý và chuyển đổi giữa 3 thiết kế kiến trúc khác nhau
"""

from typing import Dict, Any, Literal
import asyncio
import time

from .design_1_hierarchical import HierarchicalPricePredictionSystem
from .design_2_round_robin import RoundRobinPricePredictionSystem  
from .design_3_ensemble_voting import EnsembleVotingPricePredictionSystem

ArchitectureType = Literal["hierarchical", "round_robin", "ensemble_voting"]

class ArchitectureManager:
    def __init__(self, vn_api, gemini_api_key: str):
        self.vn_api = vn_api
        self.gemini_api_key = gemini_api_key
        
        # Initialize all 3 architectures
        self.architectures = {
            "hierarchical": HierarchicalPricePredictionSystem(vn_api, gemini_api_key),
            "round_robin": RoundRobinPricePredictionSystem(vn_api, gemini_api_key),
            "ensemble_voting": EnsembleVotingPricePredictionSystem(vn_api, gemini_api_key)
        }
        
        # Performance tracking
        self.performance_history = {
            "hierarchical": [],
            "round_robin": [],
            "ensemble_voting": []
        }
    
    async def predict_price(
        self, 
        symbol: str, 
        architecture: ArchitectureType = "ensemble_voting",
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        Dự đoán giá cổ phiếu sử dụng kiến trúc được chỉ định
        """
        
        if architecture not in self.architectures:
            raise ValueError(f"Architecture '{architecture}' not supported. Choose from: {list(self.architectures.keys())}")
        
        start_time = time.time()
        
        try:
            # Execute prediction with selected architecture
            system = self.architectures[architecture]
            result = await system.predict_price(symbol, timeframe)
            
            execution_time = time.time() - start_time
            
            # Add performance metadata
            result.update({
                'execution_time': execution_time,
                'architecture_used': architecture,
                'timestamp': time.time()
            })
            
            # Track performance
            self.performance_history[architecture].append({
                'symbol': symbol,
                'execution_time': execution_time,
                'confidence': result.get('confidence', 0),
                'timestamp': time.time()
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                'final_price': 0,
                'confidence': 0,
                'analysis': f'Architecture {architecture} failed: {str(e)}',
                'recommendation': 'HOLD',
                'symbol': symbol,
                'timeframe': timeframe,
                'architecture': architecture,
                'execution_time': execution_time,
                'error': str(e)
            }
    
    async def compare_architectures(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """
        So sánh kết quả của cả 3 kiến trúc
        """
        
        results = {}
        
        # Run all architectures in parallel
        tasks = [
            self.predict_price(symbol, arch, timeframe) 
            for arch in self.architectures.keys()
        ]
        
        architecture_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        for i, arch_name in enumerate(self.architectures.keys()):
            result = architecture_results[i]
            if isinstance(result, Exception):
                results[arch_name] = {
                    'error': str(result),
                    'final_price': 0,
                    'confidence': 0
                }
            else:
                results[arch_name] = result
        
        # Generate comparison summary
        comparison = {
            'symbol': symbol,
            'timeframe': timeframe,
            'results': results,
            'summary': self._generate_comparison_summary(results),
            'recommendation': self._get_consensus_recommendation(results)
        }
        
        return comparison
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo tóm tắt so sánh giữa các kiến trúc"""
        
        valid_results = {k: v for k, v in results.items() if v.get('final_price', 0) > 0}
        
        if not valid_results:
            return {
                'status': 'All architectures failed',
                'price_range': {'min': 0, 'max': 0, 'avg': 0},
                'confidence_range': {'min': 0, 'max': 0, 'avg': 0}
            }
        
        prices = [v['final_price'] for v in valid_results.values()]
        confidences = [v['confidence'] for v in valid_results.values()]
        execution_times = [v.get('execution_time', 0) for v in valid_results.values()]
        
        return {
            'successful_architectures': len(valid_results),
            'price_range': {
                'min': min(prices),
                'max': max(prices),
                'avg': sum(prices) / len(prices),
                'std': (sum((p - sum(prices)/len(prices))**2 for p in prices) / len(prices))**0.5
            },
            'confidence_range': {
                'min': min(confidences),
                'max': max(confidences),
                'avg': sum(confidences) / len(confidences)
            },
            'performance': {
                'fastest': min(execution_times),
                'slowest': max(execution_times),
                'avg_time': sum(execution_times) / len(execution_times)
            },
            'agreement_level': self._calculate_agreement_level(prices)
        }
    
    def _calculate_agreement_level(self, prices: list) -> str:
        """Tính mức độ đồng thuận giữa các dự đoán"""
        
        if len(prices) < 2:
            return "INSUFFICIENT_DATA"
        
        avg_price = sum(prices) / len(prices)
        max_deviation = max(abs(p - avg_price) / avg_price for p in prices)
        
        if max_deviation < 0.02:  # < 2%
            return "HIGH_AGREEMENT"
        elif max_deviation < 0.05:  # < 5%
            return "MODERATE_AGREEMENT"
        elif max_deviation < 0.10:  # < 10%
            return "LOW_AGREEMENT"
        else:
            return "DIVERGENT_PREDICTIONS"
    
    def _get_consensus_recommendation(self, results: Dict[str, Any]) -> str:
        """Lấy khuyến nghị đồng thuận từ các kiến trúc"""
        
        recommendations = []
        for result in results.values():
            if result.get('recommendation'):
                recommendations.append(result['recommendation'])
        
        if not recommendations:
            return "HOLD"
        
        # Count votes
        buy_votes = recommendations.count('BUY') + recommendations.count('STRONG_BUY')
        sell_votes = recommendations.count('SELL') + recommendations.count('STRONG_SELL')
        hold_votes = recommendations.count('HOLD')
        
        if buy_votes > sell_votes and buy_votes > hold_votes:
            return "BUY"
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            return "SELL"
        else:
            return "HOLD"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Lấy thống kê hiệu suất của các kiến trúc"""
        
        stats = {}
        
        for arch_name, history in self.performance_history.items():
            if not history:
                stats[arch_name] = {
                    'total_predictions': 0,
                    'avg_execution_time': 0,
                    'avg_confidence': 0
                }
                continue
            
            stats[arch_name] = {
                'total_predictions': len(history),
                'avg_execution_time': sum(h['execution_time'] for h in history) / len(history),
                'avg_confidence': sum(h['confidence'] for h in history) / len(history),
                'recent_performance': history[-5:] if len(history) >= 5 else history
            }
        
        return stats
    
    def get_architecture_info(self) -> Dict[str, str]:
        """Thông tin về các kiến trúc"""
        
        return {
            "hierarchical": "Big Agent tóm tắt từ 6 agents con - Tốt cho phân tích tổng hợp",
            "round_robin": "6 agents chạy tuần tự, cải thiện dần - Tốt cho phân tích từng bước",
            "ensemble_voting": "6 agents song song + Bayesian inference - Tốt cho độ chính xác cao"
        }