import React from "react";
import { View, Text } from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { RootStackParamList } from "../App";
import { useAppStore } from "@state/store";

export type InsightsProps = NativeStackScreenProps<RootStackParamList, "Insights">;

const InsightsScreen: React.FC<InsightsProps> = () => {
  const dailySummary = useAppStore((s) => s.dailySummary);

  const total = dailySummary?.totalScreenTime ?? 0;
  const sessions = dailySummary?.sessionCount ?? 0;

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 20, fontWeight: "bold", marginBottom: 12 }}>Insights</Text>
      <Text>Your most used app today: {dailySummary?.topApps[0] ?? "-"}</Text>
      <Text>You picked up your phone approximately {sessions} times.</Text>
      <Text>Your average session lasted {sessions ? (total / sessions).toFixed(1) : 0} minutes.</Text>
      <Text>Peak usage hour: {dailySummary?.mostCommonHour ?? 0}:00</Text>
    </View>
  );
};

export default InsightsScreen;
