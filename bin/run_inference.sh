#  12b -> 180 mins,  8 batch size -> done
# 6.9b -> 120 mins, 12 batch size -> done
# 2.8b ->  80 mins, 16 batch size -> done
# 1.4b ->  40 mins, 32 batch size -> done
# 410m ->  30 mins, 40 batch size -> done
# 160m ->  30 mins, 40 batch size -> done
#  70m ->  10 mins, 44 batch size -> done
poetry run python ./scripts/main.py -m \
	model_size=12b \
	batch_size=8 \
	+launcher=slurm \
	hydra.launcher.timeout_min=180 \
	hydra.launcher.array_parallelism=10 \
	revision=step143000,step142000,step141000,step140000,step139000,step138000,step137000,step136000,step135000,step134000,step133000,step132000,step131000,step130000,step129000,step128000,step127000,step126000,step125000,step124000,step123000,step122000,step121000,step120000,step119000,step118000,step117000,step116000,step115000,step114000,step113000,step112000,step111000,step110000,step109000,step108000,step107000,step106000,step105000,step104000,step103000,step102000,step101000,step100000,step99000,step98000,step97000,step96000,step95000,step94000,step93000,step92000,step91000,step90000,step89000,step88000,step87000,step86000,step85000,step84000,step83000,step82000,step81000,step80000,step79000,step78000,step77000,step76000,step75000,step74000,step73000,step72000,step71000,step70000,step69000,step68000,step67000,step66000,step65000,step64000,step63000,step62000,step61000,step60000,step59000,step58000,step57000,step56000,step55000,step54000,step53000,step52000,step51000,step50000,step49000,step48000,step47000,step46000,step45000,step44000,step43000,step42000,step41000,step40000,step39000,step38000,step37000,step36000,step35000,step34000,step33000,step32000,step31000,step30000,step29000,step28000,step27000,step26000,step25000,step24000,step23000,step22000,step21000,step20000,step19000,step18000,step17000,step16000,step15000,step14000,step13000,step12000,step11000,step10000,step9000,step8000,step7000,step6000,step5000,step4000,step3000,step2000,step1000,step0
