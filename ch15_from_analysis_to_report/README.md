# Chapter 15: From Analysis to Report

**What this chapter's code must prove:** plot_huber_iqr (never explicitly taught despite being central to a full day of earlier robustkit fixes) and its three export variants produce the same underlying analysis in forms suited to different audiences -- and a direction mistake in outlier flagging fails silently, not loudly.

## Status
Fully verified end-to-end.

Delivery-time dataset: 3 warehouses x 2 shifts, real route-length
relationship, real night-shift penalty, 6 genuine injected breakdowns
(+90 min, unrelated to route length/shift).

Verified:
- plot_huber_iqr(show_points=False/True): same fit, same R^2/MAE/RMSE,
  visually confirmed side-by-side difference (publisher_vs_analyst.png)
- export_huber_iqr_pdf: 6 pages (one per warehouse-shift segment)
- export_huber_iqr_images: 6 individual image files, same content
- mad_outlier_report with direction="negative" (default): 0/6
  breakdowns caught -- confirmed this is NOT an error, just an empty
  result, since all 6 breakdowns are unusually SLOW not fast
- Same call with direction="positive": 6/6 breakdowns caught (+1
  reasonable false positive from ordinary noise)
- export_outlier_pdf: 6 pages, flagged breakdown visually confirmed
  sitting well above the trend line, labeled by route_id

Chart delivery_outliers_sample.png is a real rasterized page from the
generated PDF (East_Day segment, route R1428 flagged), not a separately
redrawn chart.
