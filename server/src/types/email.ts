export interface EmailRecord {
  id?: string;
  content: string;
  prediction: "spam" | "not_spam";
  confidence?: number;
  createdAt?: Date;
  updatedAt?: Date;
}