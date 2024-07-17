from UltraDict import UltraDict

from insight.zero.component.insight_comp import InsightComponent
from simple_http.simple_http_comp import SimpleHttpComponent

clean_list = ['global', SimpleHttpComponent.SHARED_MEMORY_NAME, InsightComponent.SHARED_MEMORY_NAME]

for i in range(len(clean_list)):
    try:
        UltraDict.unlink_by_name(clean_list[i])
    except Exception as e:
        pass

