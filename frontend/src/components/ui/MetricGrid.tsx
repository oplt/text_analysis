import { ResponsiveCardGrid, type ResponsiveCardColumns } from "./ResponsiveCardGrid";

type MetricGridProps = {
    children: React.ReactNode;
    /** KPI cards: default 4 on desktop. */
    columns?: ResponsiveCardColumns;
};

/** Layout wrapper for StatCard / KPI rows. */
export function MetricGrid({ children, columns = 4 }: MetricGridProps) {
    return <ResponsiveCardGrid columns={columns}>{children}</ResponsiveCardGrid>;
}
