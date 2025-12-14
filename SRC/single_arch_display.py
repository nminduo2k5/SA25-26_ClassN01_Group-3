"""
Single Architecture Display Logic
"""

def display_single_architecture_result(st, main_agent, selected_architecture, architecture_options, symbol):
    """Display single architecture prediction result"""
    
    # Run single architecture
    st.markdown("---")
    arch_info = main_agent.single_architecture_runner.get_architecture_info(selected_architecture)
    st.markdown(f"### {architecture_options[selected_architecture]}")
    
    with st.spinner(f"Chạy {arch_info['name']}..."):
        single_result = main_agent._safe_get_single_architecture_prediction(selected_architecture, symbol, {})
    
    if single_result.get('error'):
        st.error(f"❌ {single_result['error']}")
    else:
        # Show architecture info
        st.info(f"📊 **Mô tả:** {arch_info['description']}")
        st.success(f"✅ **Tốt nhất cho:** {arch_info['best_for']}")
        
        # Display predictions
        arch_preds = single_result.get('predictions', {})
        if arch_preds:
            col1, col2, col3 = st.columns(3)
            
            for i, (timeframe, pred) in enumerate(arch_preds.items()):
                with [col1, col2, col3][i % 3]:
                    if pred.get('price'):
                        st.metric(
                            f"{timeframe.replace('_', ' ').title()}",
                            f"{pred['price']:,.2f} VND",
                            f"{pred.get('confidence', 0):.1f}% confidence"
                        )
        
        # Show additional info
        if single_result.get('participating_agents'):
            st.write(f"🤖 **Agents tham gia:** {', '.join(single_result['participating_agents'])}")
        
        method = single_result.get('method', selected_architecture)
        st.caption(f"📊 Phương pháp: {method}")