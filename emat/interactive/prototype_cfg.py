

from ..scope.box import Box, Boxes


sample_pref_meas = {
	'Region-wide VMT',
	'AM Trip Time (minutes)',
	'AM Trip Length (miles)',
	'Total Transit Boardings',
	'Total LRT Boardings',
	'Peak Transit Share',
	'Downtown to Airport Travel Time',
	'Corridor Kensington Daily VHT',
}


_prototype_universe = Boxes(
	Box('Kensington Off & Low Pop',
		allowed={'Kensington Decommissioning':[False]},
		upper_bounds={'Households and Employment in downtown':1.1},
		relevant=sample_pref_meas,
		),

	Box('LRT Strategy Off',
		parent='Kensington Off & Low Pop',
		allowed={'LRT Extension':[False]}),
	Box('LRT Strategy On',
		parent='Kensington Off & Low Pop',
		allowed={'LRT Extension': [True]}),

	Box('High Population Growth',
		lower_bounds={'Households and Employment in downtown':1.0},
		relevant=sample_pref_meas),


	Box('Kensington Strategy On',
		parent='High Population Growth',
		allowed={'Kensington Decommissioning': [True]}),
	Box('Kensington Strategy Off',
		parent='High Population Growth',
		allowed={'Kensington Decommissioning': [False]}),

)

#
# _prototype_universe['High Population Growth Cluster'].uncertainty_thresholds = {
# 	'Households and Employment in downtown':(1.0, None),
# }
#
#
# _prototype_universe['Kensington Strategy Off'].lever_thresholds = {
# 	'Kensington Decommissioning':'Off',
# }
#
# _prototype_universe['Kensington Strategy On'].lever_thresholds = {
# 	'Kensington Decommissioning':'On',
# }
#
#
# _prototype_universe['Kensington Off & Low Pop Cluster'].uncertainty_thresholds = {
# 	'Households and Employment in downtown':(None, 1.1),
# }
#
# _prototype_universe['Kensington Off & Low Pop Cluster'].lever_thresholds = {
# 	'Kensington Decommissioning':'Off',
# }
#
# _prototype_universe['LRT Strategy Off Cluster'].lever_thresholds = {
# 	'LRT Extension':'Off',
# }
#
# _prototype_universe['LRT Strategy On Cluster'].lever_thresholds = {
# 	'LRT Extension':'On',
# }
#





# _prototype_universe['High Population Growth Cluster'].relevant_features |= sample_pref_meas
# _prototype_universe['Kensington Off & Low Pop Cluster'].relevant_features |= sample_pref_meas
